# Leaf Disease Prediction using GPDCNN

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This project aimed at helping farmers identify and diagnose plant diseases through image analysis. This project utilizes the GPDCNN (Global Pooling Dilated Convolutional Neural Network) architecture, which employs dilated convolutional layers with global pooling. The model is initialized with pre-trained weights from the AlexNet architecture.

<p align="center">
  <img src="https://github.com/ali0salimi/cucumber-leaf-disease-prediction/blob/main/dataset_sample.png" alt="Project Demo" width="800">
</p>

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Reference](#reference)
- [Contributing](#contributing)
- [License](#license)

## Features

- Accurate diagnosis of plant diseases using advanced convolutional neural networks.
- Efficient architecture leveraging dilated convolution and global pooling techniques.
- Pre-trained weights from the AlexNet model for improved initialization.
- User-friendly interface for inputting plant images and receiving disease predictions.

## Installation

1. Clone the repository:
   
   ```sh
   git clone https://github.com/your-username/cucumber-leaf-disease-prediction.git
   cd cucumber-leaf-disease-prediction.git
   ```
   
## Usage

1. Download GPDCNN.py

2. Install requirments
   ```sh
   pip install -r requirements.txt
   ```
3. Run the following code
   ```sh
   from GPDCNN import GPDCNN
   predictor = GPDCNN()
   predictor.predict(PATH_TO_CUCUMBER_LEAF_IMAGE)
   ```

## Dataset
[Link to dataset](https://data.mendeley.com/datasets/y6d3z6f8z9/1)

1. **Number of Classes**: The dataset comprises eight distinct cucumber disease classes: Anthracnose, Bacterial Wilt, Belly Rot, Downy Mildew, Pythium Fruit Rot, Gummy Stem Blight, Fresh leaves (healthy leaves), and Fresh cucumber (healthy cucumbers) in thi project we only used Image related to leafs which are Anthracnose, Bacterial Wilt, Downy Mildew, Gummy Stem Blight, Fresh leaves.

2. **Data Collection**: A total of 1280 original images of cucumber samples were collected directly from real fields. These images serve as the foundation for building the dataset.

3. **Data Augmentation**: To enhance the diversity and size of the dataset, data augmentation techniques have been applied to the original images. These augmentation methods include flipping, shearing, zooming, and rotation. As a result, a total of 6400 augmented images were generated from the original 1280 images.


## Model Architecture

<p align="center">
  <img src="https://github.com/ali0salimi/cucumber-leaf-disease-prediction/blob/main/model-architecture.png" alt="Project Demo" width="800">
</p>
GPDCNN, a modified CNN inspired by AlexNet, employs dilated convolutions and global pooling. With 13 layers including Inception and Concat, it enhances feature extraction for detecting cucumber leaf diseases. Dilated convolutions capture intricate patterns, while global pooling aids generalization, fostering accurate disease identification.

## Results

accuracy on test set after 30 epochs 
```sh
16/16 [==============================] - 5s 116ms/step - loss: 0.1125 - accuracy: 0.9688
accuracy on test set : 0.96875
```

## Reference

If you're using the GPDCNN architecture, it's based on the research described in the following paper:

- Author(s). <i>Cucumber leaf disease identiﬁcation with global pooling dilated
convolutional neural network<i> *Elsevier*, 2019. [Link to the Paper](https://www.sciencedirect.com/science/article/abs/pii/S0168169918317976)

Please make sure to give proper credit to the original authors by referencing their work.

## Contributing

Contributions are welcome! If you find any issues or want to enhance the project, feel free to submit a pull request.

