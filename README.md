<p align="center">
  <img src="https://www.science.org/do/10.1126/science.aac4663/abs/sn-rockpaper.jpg" alt="Rock, Paper, Scissors Classifier" width="300">
</p>

# Rock, Paper, Scissors Classifier
This repository contains a Python script that builds and trains a state-of-the-art convolutional neural network (CNN) to classify images of rock, paper, and scissors. The CNN is implemented using TensorFlow and Keras.

___

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

___

## Requirements
To run the script and train the model, you need to install the following Python packages:
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Requests](https://docs.python-requests.org/en/master/)
- [scikit-learn](https://scikit-learn.org/stable/)
  
You can install these packages using the provided `requirements.txt` file. See the [Installation](#installation) section for details.

___

## Usage
Follow the steps below to use this repository:

### 1. Clone the Repository
Clone this repository to your local machine using the following command:
```bash
git clone https://github.com/your-username/rock-paper-scissors.git
```

### 2. Install Requirements
Navigate to the cloned repository and install the required Python packages using pip:
```bash
pip install -r requirements.txt
```
### 3. Download the Dataset
The dataset used in this project can be downloaded from Kaggle. Once downloaded, place the dataset in the rockpaperscissors directory.

### 4. Train the Model
Run the script to train the CNN model:
```python
pyhton app-name.py
```
### 5. Evaluate the Model
After training, the script will evaluate the model on the validation set and display the accuracy. You can also evaluate the model on the training set by uncommenting the relevant code in train_model.py.

### 6. Make Predictions
You can use the trained model to make predictions on new images by replacing 'path_to_image.jpg' with the path to your image file in the provided code.

___

### Dataset
The dataset consists of thousands of high-quality images of rock, paper, and scissors. It has been preprocessed and organized into training and validation sets, with data augmentation applied during training to enhance model robustness.

### Model
The CNN model is a deep neural network architecture that leverages convolutional layers, max-pooling layers, and fully connected layers. It has been meticulously fine-tuned to achieve outstanding performance in classifying images into one of three classes: rock, paper, or scissors.

### Evaluation
The model's performance is evaluated on both the training and validation sets, and the accuracy is displayed, showcasing its capability to achieve remarkable results.

### Prediction
Utilize the provided code to harness the power of this trained model for making predictions on new images. Replace 'path_to_image.jpg' with the path to your image file, and watch the model expertly predict the class (rock, paper, or scissors).

### Contributing
Contributions are welcome! Feel free to open issues, submit pull requests, or suggest enhancements. Let's collaborate to make this project even better.

### License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/rzqllh/RPS-Detection_WINDOWS/blob/main/LICENSE) file for details.
