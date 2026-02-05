# Computer Vision Deep Learning License Plate Reader Project

This project is a recognition system using deep learning to detect and read license plates in real-time. This system will integrate multiple
stages of computer vision processing, such as vehicle detection, license plate localization, object tracking, and optical character recognition, 
being done in a modular and fully automated pipeline via Python. By combining cutting-edge tools such as YOLOv11 for object detection, SORT for 
vehicle tracking, and EasyOCR for text recognition, the project aims to simulate a real-world license plate reader capable of working with video 
footage from busy roads or surveillance cameras.

The license plate reader project will implement many topics from 3D Image
Processing and Computer Vision, particularly the transition from traditional computer
vision methods like edge detection and feature matching to modern deep learning
approaches. 

Concepts such as feature extraction using convolutional neural networks
(CNNs), bounding box regression, object classification, and Kalman filtering for motion
tracking were all explored in lectures and applied practically in this implementation. The
integration of these components illustrates how real-world systems are built using layered
vision models and machine learning inference techniques.

#### Table of Contents

- [Important Links](#Important-Links)
- [Model Summary](#Model-Summary)
- [Results](#Results)
- [References](#References)
- [Installation & Setup](#Installation--Setup)

### Important Links

### Model Summary

### Results

#### Training Results for YOLOv11

Training the YOLO model took around 2 hours, due to the large size of the dataset and the
number of epochs specified. The key hyperparameters used to train the model are shown:
| Hyperparamter     | Setting | 
| :---------------- | :------: |
| Number of Epochs  |   100    | 
| Batch Size        |   16     | 
| Learning Rate     |  0.01    | 
| Momentum          |  0.9     | 



### References   

### Installation & Setup
