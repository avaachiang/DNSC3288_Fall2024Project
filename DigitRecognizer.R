library(reticulate)
use_condaenv("r-tensorflow", required = TRUE)
py_config()

library(tensorflow)
install_tensorflow()

# Install libraries and set up Keras and TensorFlow
install.packages(c("keras", "tensorflow", "ggplot2", "caret", "data.table"))
library(keras)
install_keras()  # Installs Keras and TensorFlow

# Download datasets
library(data.table)
train <- fread("train.csv")
test <- fread("test.csv")

# Split data to eparate the features (images) and labels (digits)
train_labels <- train$label
train_images <- as.matrix(train[, -"label", with = FALSE])
test_images <- as.matrix(test)

# Normalize the pixel values to [0, 1]
train_images <- train_images / 255
test_images <- test_images / 255

# Reshape the data to fit Keras' input requirements (28x28x1 for grayscale images)
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# One-hot encode the labels
train_labels <- to_categorical(train_labels, num_classes = 10)

# Build the neural network model (CNN)
# Define the model 
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

# Compile the model
model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Train the model 
history <- model %>% fit(
  train_images, train_labels,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

# Predict probabilities
predictions_probabilities <- model %>% predict(test_images)

# Get predicted classes (0-9) from probabilities
predictions <- k_argmax(predictions_probabilities) %>% as.array()

# Create submission file
submission <- data.table(ImageId = 1:nrow(test_images), Label = predictions)

# Save the submission file as a CSV
fwrite(submission, "submission.csv")

# Visualize training history
plot(history)

# Display sample predictions for each digit (0 to 9)
library(ggplot2)

# Set up a 2x5 grid for displaying digits
par(mfrow = c(2, 5), mar = c(1, 1, 2, 1), oma = c(0, 0, 2, 0)) 


# Loop through each digit (0 to 9)
for (digit in 0:9) {
  # Find the first index where the predicted label matches the current digit
  index <- which(predictions == digit)[1]  # Take the first instance
  
  if (!is.na(index)) {
    # Plot the corresponding image
    plot(as.raster(test_images[index, , , 1], max = 1))
    
    # Add the predicted label as the title
    title(main = paste("Predicted:", digit), cex.main = 1.5)
  } else {
    # If no instance of the digit is found, create a blank placeholder
    plot.new()
    title(main = paste("Missing:", digit), cex.main = 1.5)
  }
}

# Reset plotting layout
par(mfrow = c(1, 1))

# Average Accuracy for training dataset
# Predict the probabilities for the training dataset
train_predictions_probabilities <- model %>% predict(train_images)

# Convert probabilities to predicted classes
train_predictions <- k_argmax(train_predictions_probabilities) %>% as.array()

# Extract true labels for training
train_true_labels <- apply(train_labels, 1, which.max) - 1  # Convert one-hot encoding back to numeric labels

# Calculate training accuracy
train_accuracy <- mean(train_predictions == train_true_labels)
print(paste("Training Accuracy:", round(train_accuracy * 100, 2), "%"))

# Average Accuracy for validation dataset
# Extract validation data (since validation_split = 0.2 during training)
validation_split_index <- floor(0.8 * nrow(train_images))  # 80% for training
validation_images <- train_images[(validation_split_index + 1):nrow(train_images), , , ]
validation_labels <- train_labels[(validation_split_index + 1):nrow(train_labels), ]

# Predict the probabilities for the validation dataset
validation_images <- array_reshape(validation_images, c(nrow(validation_images), 28, 28, 1))
validation_predictions_probabilities <- model %>% predict(validation_images)

# Convert probabilities to predicted classes
validation_predictions <- k_argmax(validation_predictions_probabilities) %>% as.array()

# Extract true labels for validation
validation_true_labels <- apply(validation_labels, 1, which.max) - 1  # Convert one-hot encoding back to numeric labels

# Calculate validation accuracy
validation_accuracy <- mean(validation_predictions == validation_true_labels)
print(paste("Validation Accuracy:", round(validation_accuracy * 100, 2), "%"))

# Calculating AUC - UNUSED
# Load necessary library
install.packages("pROC")  # Install if not already installed
library(pROC)

# Predict probabilities for the training data (to calculate AUC for training set)
train_probabilities <- model %>% predict(train_images)

# Extract true labels for the training set (convert one-hot encoded labels to numeric class indices)
train_true_labels <- apply(train_labels, 1, which.max) - 1  # Convert one-hot back to original labels
colnames(train_probabilities) <- 0:9

# Calculate AUC for the training dataset
train_auc <- multiclass.roc(train_true_labels, train_probabilities)
train_auc_value <- auc(train_auc)
print(paste("Training AUC:", train_auc_value))

# During training, validation_split = 0.2 means 20% of the data was used for validation.
# Extract validation data from the training dataset
validation_split_index <- floor(0.8 * nrow(train_images))
validation_images <- train_images[(validation_split_index + 1):nrow(train_images), , , ]
validation_labels <- train_labels[(validation_split_index + 1):nrow(train_labels), ]
validation_images <- array_reshape(validation_images, c(nrow(validation_images), 28, 28, 1))

# Predict probabilities for the validation data
validation_probabilities <- model %>% predict(validation_images)

# Extract true labels for the validation set
validation_true_labels <- apply(validation_labels, 1, which.max) - 1  # Convert one-hot encoding to numeric labels
validation_true_labels <- as.numeric(as.factor(validation_true_labels)) - 1
colnames(validation_probabilities) <- 0:9

# Calculate AUC for the validation dataset
validation_auc <- multiclass.roc(validation_true_labels, validation_probabilities)
validation_auc_value <- auc(validation_auc)
print(paste("Validation AUC:", validation_auc_value))
