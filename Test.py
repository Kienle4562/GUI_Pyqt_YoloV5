# import cv2
# # Open the device at the ID 0
# cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
# # Check whether user selected camera is opened successfully.
# if not (cap.isOpened()):
#     print("Could not open video device")
# else:
#     print("ok")
from PIL import Image
import cv2
import torch
import math
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper

# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True, help='path to input image')
# args = ap.parse_args()

from Yolo_GUI import detect_lp

detect_lp(filename="1")