from ultralytics import YOLO
import cv2

from sort.sort import *
from util import get_car, read_license_plate, write_csv
from config import video, output_csv

# for writing to csv and saving info
results = {}

# object tracker used to track all the vehicles in the video
mot_tracker = Sort()


