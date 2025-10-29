import string
import easyocr
import re

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    pass


