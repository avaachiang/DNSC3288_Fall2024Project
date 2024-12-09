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

# Build the neural network (CNN)
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

# Display sample predictions
library(ggplot2)
plot(as.raster(test_images[1, , , 1], max = 1))
print(paste("Predicted label:", predictions[1]))


















