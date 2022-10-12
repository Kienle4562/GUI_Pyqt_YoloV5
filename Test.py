import cv2
# Open the device at the ID 0
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
# Check whether user selected camera is opened successfully.
if not (cap.isOpened()):
    print("Could not open video device")
else:
    print("ok")