"""License plate recognition for video.

The pipeline detects vehicles and plates with YOLO11, follows vehicles with a
SORT tracker, reads each plate crop with an OCR backend, and combines the reads
of a track into one plate string.
"""

__version__ = "1.0.0"
