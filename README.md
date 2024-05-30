# Autonomous-Drone-Line-Follower

This repository contains the source code and documentation for an autonomous drone line-following project. The project aims to leverage deep learning techniques to enable a drone to autonomously follow a predefined path.

## Getting Started
### To replicate this project, follow these steps:
1. Click the green “Code” button in the top right corner.
2. Select “Download ZIP” to download the repository as a compressed ZIP file.
3. Extract the contents of the ZIP to a folder on your computer.
4. Ensure Python 3.10 is installed on your computer.
5. (Optional) If utilizing an RTX 4060 GPU, use Anaconda to set up a virtual environment and install required GPU libraries for CUDA(11.2) and cuDNN(8.1).
6. (Optional) If encountering a ptax error, use the command conda install -c nvidia cuda-nvcc.
7. Install the required packages using pip install -r requirements.txt in the Anaconda/normal command prompt in the extracted folder.
8. Ensure relevant images, models, etc., are installed as GitHub has a file size limit.

## Features
1. Integration of CNN, VGG16, and U-Net models for image processing and path detection.
2. Real-time object detection and image segmentation using pre-trained neural network models.
3. Implementation of TensorFlow, OpenCV, and Keras for model training and evaluation.
4. Integration with DJI Tello drone for live video feed and real-world testing.

## Objectives
- Develop neural network models capable of recognizing and following designated paths in real-time.
- Integrate these models with drone hardware to enable autonomous navigation.
- Conduct extensive testing and analysis to evaluate the performance and effectiveness of the developed models.
- Identify and address challenges, such as drone drifting, to improve overall system functionality.

## Methodology
1. The project employs three main neural network architectures: conventional CNN, VGG16, and U-Net.
2. Data preprocessing techniques are applied to prepare images for training and testing.
3. Model training involves adjusting parameters such as batch size, epochs, and validation split to optimize performance.
4. Testing is conducted using both simulated scenarios and real-world drone feeds.
5. Performance analysis includes evaluating accuracy, loss, and computational efficiency.
6. The project also explores image transformation techniques and visualizes neural network architectures for better understanding.

### Control Logic of Drone Camera
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/66f069d8-3557-4555-b7a6-9be5ebb88863)

### Model Implementation Workflow Overview
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/fa725946-b222-4030-afe4-7093520d301e)

### Drone Integration
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/4110ab4a-7550-4a9c-b439-b328a43809ef)

## Implementation

### Traditional CNN Model
- Python script: mainGPU.py
- Description: Trains a traditional CNN model using newtrain.py.
- Model architecture: Sequential with layers for convolution, batch normalization, max pooling, flattening, dense, and dropout.
- Training process: Logs training and validation metrics to a CSV file.

#### MainGPU Architecture Visualized Output
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/91e78986-20e5-4632-879f-669586d30254)

### VGG16 Model
- Python script: VGG16GPU.py
- Description: Trains a VGG16 model for object detection.
- Data preprocessing: Images resized and augmented using Keras's ImageDataGenerator.
- Training process: Utilizes pre-trained VGG16 base model with additional layers for bounding box prediction.

#### VGG16 Architecture Visualized Output
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/932047a9-05da-45c8-83d8-1ba64cf32421)

### U-Net Model
- Python script: mainUNET.py
- Description: Implements U-Net architecture for image segmentation.
- Data preprocessing: Converts images to grayscale, applies edge detection, and generates segmentation masks.
- Model architecture: Contracting path, bottleneck, and expansive path with convolutional, ReLU, max pooling, dropout, and upsampling layers.

#### U-Net Architecture Visualized Output
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/36e72b72-835b-4ac5-be6f-c52123caccda)

## Testing

### Traditional CNN Model:
Python script: mainTest.py Description: Performs object detection on a series of images using a pre-trained CNN model.

### VGG16 Model:
Python script: vgg16TestWithDrone.py Description: Uses Tello drone camera to capture video frames, process them, and display them in real-time with bounding boxes drawn based on predictions from a pre-trained VGG16 model.

### U-Net Model:
Python script: unetTestWithDrone.py Description: Utilizes a pre-trained U-Net model to process live video feed from a Tello drone for real-time image segmentation.

### Image Transformation Techniques
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/f57b5438-eda7-4c33-b32a-ea69297d37a2)

### Dataset Image under Edge Detection
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/2581a343-8ec4-4e7b-b014-279b181d75d9)

### VGG16 Model Line Tracking With Bounding Boxes
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/5fcfc7f9-72b3-4e00-b898-30f75b78e67c)

### U-Net Image Segmentation Process
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/b07d6da6-b662-457f-91ec-ca32d79e80e3)

### U-Net Image Labelling Process
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/ee73d956-2950-4042-b255-10ef120c1691)

## Performance Measurement Results
1. The project evaluates the performance of each model based on accuracy, loss, and computational efficiency.
2. Graphical representations and plots are used to visualize training and testing results.
3. Analysis indicates that the U-Net model outperforms the conventional CNN and VGG16 models in terms of accuracy and loss.
4. Parameters such as batch size, epochs, and validation split are optimized to enhance model performance.

### Performance of Convolutional Neural Network (CNN) Model
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/d35f6e3b-1662-4557-a90b-eec2ab833a68)

### Performance of VGG16 Model
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/d081dc32-a441-4e34-a077-f755174df9b8)

### Performance of U-Net Model
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/625f60e8-ef71-47ab-8c94-80d34126bef5)

## Enhancing Neural Network Models for Precision and Innovation of Autonomous Line Tracking - Scholarly Article 
![image](https://github.com/HamzaIqbal22/Autonomous-Drone-Line-Follower/assets/81776951/7cca971d-d98d-42e6-aa1b-919d983ec50d)


