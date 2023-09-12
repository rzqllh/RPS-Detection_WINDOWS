# Rock, Paper, Scissors Image Classifier

<p align="center">
  <span>
   <img src="https://www.science.org/do/10.1126/science.aac4663/abs/sn-rockpaper.jpg" alt="Rock, Paper, Scissors" height="300">
    <br>
  Image credit: Yiap See fat
  </span>
</p>

Welcome to the world of Rock, Paper, Scissors—reimagined through the lens of deep learning! This exciting project combines the nostalgia of a classic hand game with the cutting-edge power of artificial intelligence. Whether you're a machine learning enthusiast or simply a fan of the timeless Rock, Paper, Scissors, this project promises an engaging and interactive experience.

## Table of Contents

- [Introduction](#introduction)
- [About](#about)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating Model Performance](#evaluating-model-performance)
- [Making Predictions](#making-predictions)
- [Kaggle Dataset](#kaggle-dataset)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Welcome to the world of Rock, Paper, Scissors—reimagined through the lens of deep learning! This exciting project combines the nostalgia of a classic hand game with the cutting-edge power of artificial intelligence. Whether you're a machine learning enthusiast or simply a fan of the timeless Rock, Paper, Scissors, this project promises an engaging and interactive experience.

## About

This project leverages the power of TensorFlow and Keras to train a convolutional neural network (CNN) that can classify images of rock, paper, and scissors hand signs. It's a fantastic way to delve into the world of deep learning while having fun with a popular game.

## Getting Started

### Requirements

Before diving into the world of Rock, Paper, Scissors AI, make sure you have the following prerequisites installed:

- [TensorFlow](https://www.tensorflow.org/)
- [Numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Requests](https://pypi.org/project/requests/)
- [Scikit-learn](https://scikit-learn.org/stable/)

You can easily install these dependencies by running:

```shell
pip install -r requirements.txt
```

### Installation

Get started in three simple steps:

1. Clone the repository:
   ```shell
   git clone https://github.com/yourusername/rock-paper-scissors-classifier.git
   cd rock-paper-scissors-classifier
   ```
2. Organize and prepare your dataset by following the instructions in `main.py`.
3. Train the model:
   ```python
   python main.py
   ```

## Usage

### Training the Model

Train your Rock, Paper, Scissors classifier by running `main.py`. This script will download the dataset, organize it into training and validation sets, and train a deep learning model for gesture recognition.

```shell
python main.py
```

### Evaluating Model Performance

After training, evaluate the model's performance on both the training and validation sets. This will help you gauge how well the AI can play Rock, Paper, Scissors.

## Making Predictions

To make predictions on new images, use the predict_image function provided in predict.py. Replace 'path_to_image.jpg' with the path to your image file.

```shell
python predict.py
```

## Kaggle Dataset

The dataset for this project was obtained from Kaggle. You can find the original dataset and additional resources on the [Kaggle Rock, Paper, Scissors Dataset](https://www.kaggle.com/drgfreeman/rockpaperscissors) page. The dataset includes a variety of hand gesture images, making it a great starting point for training your image classifier.

## Contributing

We welcome contributions from the community. Whether it's improving the model, enhancing the documentation, or adding new features, your contributions can make this project even more exciting. Feel free to open issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/rzqllh/RPS-Detection_WINDOWS/blob/main/LICENSE) file for details.
