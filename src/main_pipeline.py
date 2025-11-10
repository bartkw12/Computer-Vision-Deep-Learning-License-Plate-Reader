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

# class id for car, motorbike, bus, and truck
vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1

ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # 1 detect vehicles
        detections = coco_model(frame)[0]

        # 2 save all bounding boxes of vehicles we detect in video
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # 3 track vehicles - object tracking
        track_ids = mot_tracker.update(np.asarray(detections_))  # all BB of all vehicles detected w tracking info

        # 4 detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            # call get car function from util.py to return the car belonging to license plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]



