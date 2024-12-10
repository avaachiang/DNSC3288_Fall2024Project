# DNSC3288_Fall2024Project

### Basic Information
* **Person or organization developing model**: Ava Chiang, `avaachiang@gwu.edu`
* **Model date**: December, 2024
* **Model version**: 1.0
* **License**: Apache 2.0
* **Model implementation code**: [DigitRecognizer.R](https://github.com/avaachiang/DNSC3288_Fall2024Project/blob/main/DigitRecognizer.R)

### Intended Use
* **Primary intended uses**: This model is intended to be used as the semester project (Fall 2024) for DNSC 3288 Big Data, Predictive Analytics, and Ethics taught by Professor Patrick Hall. It is a submission for the Digit Recognizer competition on Kaggle to practice preprocessing image data, training a Convolutional Neural Network (CNN) model, and generating predictions for a test dataset. It is evaluated by its prediction accuracy. 
* **Primary intended users**: Professor Patrick Hall 
* **Out-of-scope use cases**: Any use beyond academic and learning purposes is out of scope. 

### Training Data
* Data dictionary: 

| Name | Modeling Role | Measurement Level| Description|
| ---- | ------------- | ---------------- | ---------- |
|**ImageID**| ID | integer | unique row indentifier for submission file |
| **Label** | target | integer | Target variable representing the digit (0-9) in *submission.csv* |
| **label** | target | integer | Target variable representing the digit (0-9) in *train.csv* |
| **pixel0 - pixel783** | input| integer (binary) | Intensity values of pixels in a 28x28 grayscale image, flattened into a 1D vector. Each value ranges from 0 (black) to 1 (white), representing normalized pixel intensity. |

* **Source of training data**: Kaggle Competition Digit Recognizer (*train.csv*)
* **How training data was divided into training and validation data**: 48% training, 12% validation, 40% test
* **Number of rows in training and validation data**:
  * Training rows: 33,600
  * Validation rows: 8,400

### Test Data
* **Source of test data**: Kaggle Competition Digit Recognizer (*test.csv*)
* **Number of rows in test data**: 28,000
* **State any differences in columns between training and test data**: None

### Model details
* **Columns used as inputs in the final model**: 'ImageID',
       'Label', 'label', 'pixel0 - pixel783'
* **Column(s) used as target(s) in the final model**: 'Label'
* **Type of model**: Convolutional Neural Network (CNN)
* **Software used to implement the model**: R, R Packages (keras, tensorflow, data.table, ggplot2, caret, pROC), reticulate (Python functions and libraries)
* **Version of the modeling software**: R version 4.4.2, Python version 3.11.1
* **Hyperparameters or other settings of your model**: 
```
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
```
### Quantitative Analysis

* Models were assessed with Accuracy. See details below:

| Train Accuracy | Validation Accuracy | Test Accuracy |
| ------ | ------- | -------- |
| 0.9955 | 0.9882 | 0.98832* |

(*Test AUC taken from [https://github.com/avaachiang/DNSC3288_Fall2024Project/blob/main/DigitRecognizer.R] after being submitted on Kaggle Competition Digit Recognizer)

#### Training and Validation Prediction Accuracy 
![Training and Validation Prediction Accuracy](https://github.com/avaachiang/DNSC3288_Fall2024Project/blob/main/AccruacyGraph.png)
(*the value 1 on the x-axis corresponds to the Digit 0, and so on*)

#### Sample Prediction for Each Digit (0-9)
![sample predictions for each digit](https://github.com/user-attachments/assets/3fd03f3f-eddc-44ba-8199-694ea367c78d)

### Ethical Considerations
* **Describe potential negative impacts of using your model:**
   ■ Negative software impacts of this model can include overfitting and bias in data distribution. Overfitting may cause the model to fail to generalize well to unseen data, leading to poor predictions in real-world scenarios. If the training data is not representative of all possible handwritten digit styles, the model may struggle with recognizing certain handwriting patterns.
   ■ A real-world risk of this is discrimination during standardized testing if a Convolutional Neural Network (CNN) model is used to help grade written answers. The model may have trouble identifying numbers and letters accurately or fail to idenify any at all. 
* **Describe potential uncertainties relating to the impacts of using your model:**
   ■ Potential software and mathematical uncertainties relating to the impacts of this model include uncertainty in model interpretability and in metrics. The CNN model is a "black-box" model, making it difficult to explain why certain predictions were made, and evaluation metrics like accuracy might not fully capture edge cases or misclassifications, leading to overconfidence in the model's performance.
   ■ People with unique handwriting styles might not be represented in the training dataset, causing the model to perform poorly in certain instances. In addition, the model might misclassify digits in environments where lighting, image quality, or handwriting varies significantly from the training data.
* **Describe any unexpected results:**
   ■ Both training and validation accuracy were much higher than I expected. According the metric chosen (accuracy), the CNN model performed very well on the test dataset. But since accuracy across all three datasets are close to 99%, there may be an overfitting problem or imbalanced dataset. Without realistic test scenarios (ex: noisy images, separate handwriting style, etc.) it is difficult to gauge the effectiveness of this model. 







