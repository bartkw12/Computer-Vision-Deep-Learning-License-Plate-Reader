import ast

import cv2
import numpy as np
import pandas as pd

from config import video, output_csv_interpolated, output_video

# need to draw borders
def draw_border(img, top_left, bottom_right, color=(0, 171, 255), thickness=3, line_length_x=40, line_length_y=40):
    x1, y1 = top_left
    x2, y2 = bottom_right

# need to input video

