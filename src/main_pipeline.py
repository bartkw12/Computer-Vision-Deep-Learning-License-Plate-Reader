from ultralytics import YOLO
import cv2

from sort.sort import *
from util import get_car, read_license_plate, write_csv
from config import video, output_csv

# for writing to csv and saving info
results = {}

# object tracker used to track all the vehicles in the video
mot_tracker = Sort()

# load models
coco_model = YOLO('./Pretrained COCO/yolo11n.pt')                      # Car detection model
license_plate_detector = YOLO('./runs/detect/train3/weights/best.pt')  # License plate detection model

# load video
cap = cv2.VideoCapture(video)
