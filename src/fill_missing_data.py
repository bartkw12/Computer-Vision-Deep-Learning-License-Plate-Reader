import csv
import numpy as np
from scipy.interpolate import interp1d
from config import output_csv, output_csv_interpolated

def interpolate_bounding_boxes(data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
