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
- [Installation & Setup](#Installation--Setup)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [References](#References)

### Important Links

### Model Summary

### Pipeline Architecture

The system operates through the following stages:

1. Vehicle Detection  
   A pre-trained YOLO model identifies vehicles within each video frame.

2. License Plate Detection  
   A custom-trained YOLOv11 model detects license plates inside detected vehicle regions.

3. Object Tracking  
   SORT tracking assigns consistent IDs to vehicles across frames using Kalman filtering and Hungarian matching.

4. Optical Character Recognition  
   EasyOCR extracts alphanumeric characters from cropped license plate images.

5. Output Generation  
   Detected plates and recognized text are annotated and saved for analysis.

### Results

#### Training Results for YOLOv11

Training the YOLO model took around 2 hours, due to the large size of the dataset and the
number of epochs specified. The key hyperparameters used to train the model are shown:
| Hyperparameter     | Setting | 
| :---------------- | :------: |
| Number of Epochs  |   100    | 
| Batch Size        |   16     | 
| Learning Rate     |  0.01    | 
| Momentum          |  0.9     | 
| Weight Decay      |  0.0005  |
| Parameters        |  2.6M    |

Results of the training process given by YOLO:
| Metric      | Value | Interpretation | 
| :---------------- | :------: | :------: |
| Precision | 0.975 | 97.5% of predicted plates are correct |
| Recall | 0.972 | 97.2% of actual plates were correctly detected |
| mAP50 | 0.985 | Detection accuracy (IoU0.5) |
| mAP50-95 | 0.715 | Performance over stricter thresholds |
| Val Box Loss | 1.087 | Localization error |
| Val Class Loss | 0.357 | Confidence for classification |
| Val DFL Loss | 1.129  | Distance from predicted to ground truth |

After training this pipeline, I was very impressed and satisfied with the results. A very high
precision and recall would mean a high confidence score for each license plate, and an
easier job for the rest of the pipeline.

This model was just trained on license plates and not on cars. A second pre-trained model
for the COCO dataset from Ultralytics was used to detect vehicles. This model was
pretrained to detect different types of vehicles.

### Installation & Setup

#### 1. Clone Repository

```bash
git clone https://github.com/bartkw12/Computer-Vision-Deep-Learning-License-Plate-Reader.git
cd Computer-Vision-Deep-Learning-License-Plate-Reader
```

#### 2. Create Virtual Environment


```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Usage

```bash
python main_pipeline.py
```

### Future Improvements

The Toronto Police Services have implemented a license plate scanner on their vehicles
called Automatic License Plate Recognition (ALPR) technology to receive real-time alerting
in patrol vehicles [7]. This license plate scanner is much more accurate and works much
better than my implementation, meaning there is a lot of room for improvement. The
license plate readers' tracking ability could be enhanced by using deep SORT. This would
incorporate appearance features using CNN embeddings, more robust ID switching, and it
is overall better for tracking in crowded or complex scenes.

Other improvements could be made to the OCR accuracy. A custom-trained model could
be used to fine-tune OCR on license plate images. Post-OCR correction could be improved
by not only using a regex but also implementing a language model/pattern matching to fix
misread characters. 

The license plate reader project demonstrates a functional and efficient detection system
powered by deep learning that could be scaled to real-life deployment. By utilizing
YOLOv11 for object detection, SORT for tracking, and EasyOCR for text recognition, the full
pipeline was created that can process raw video footage, identify vehicles, localize license
plates, and extract and display readable text. 

The system demonstrated great results,
showing a high detection accuracy and robust detection in diverse conditions of license
plates, highlighting the strengths but also showcasing the weaknesses and areas of
improvement for this implementation.

### References   

[1] What is deep learning?. IBM. (2025, April 17). https://www.ibm.com/think/topics/deep-
learning

[2] Keita, Z. (2024, September 28). Yolo Object Detection explained: A beginner’s guide.
DataCamp. https://www.datacamp.com/blog/yolo-object-detection-explained

[3] Yolo11 new. Ultralytics YOLO Docs. (2025, February 26).
https://docs.ultralytics.com/models/yolo11/#overview

[4] Abewley. (n.d.). Abewley/Sort: Simple, online, and realtime tracking of multiple objects
in a video sequence. GitHub. https://github.com/abewley/sort

[5] Mahajan, A. (2023, October 29). EasyOCR: A comprehensive guide. Medium.
https://medium.com/@adityamahajan.work/easyocr-a-comprehensive-guide-
5ff1cb850168

[6] Projects, R. U. (2025, April 2). License Plate Recognition Dataset and pre-trained model
by Roboflow universe projects. Roboflow. https://universe.roboflow.com/roboflow-
universe-projects/license-plate-recognition-rxg4e/dataset/4/images
