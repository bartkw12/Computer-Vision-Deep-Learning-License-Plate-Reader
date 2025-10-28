import string
import easyocr
import re

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)