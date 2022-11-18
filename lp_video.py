import shutil

from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import function.helper as helper

yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='Model_Yolo/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='Model_Yolo/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60
print('Run LP',os.path.exists('images/tmp/single_org_vid.jpg'))
# while True:
with open("Sample_OutPut\Image_data_management.txt", 'r') as f:
    for count, line in enumerate(f):
        pass
    line = count + 1
    print('line', line)
    f = open("Sample_OutPut\Image_data_management.txt")
    list_content = f.readlines()
    # i1 = 0
    for path in range(line):
        # i1 = i1+1
        filepath = list_content[path]
        filepath = filepath.replace("\n", "")
        print("file:",filepath)
        if os.path.exists(filepath):
            img = cv2.imread(filepath)
            plates = yolo_LP_detect(img, size=640)
            print("RUN")
            list_plates = plates.pandas().xyxy[0].values.tolist()
            list_read_plates = []
            if len(list_plates) == 0:
                lp = helper.read_plate(yolo_license_plate,img)
                if lp != "unknown":
                    cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    list_read_plates.append(lp)
            else:
                for plate in list_plates:
                    flag = 0
                    print("plate",plate)
                    x = int(plate[0]) # xmin
                    y = int(plate[1]) # ymin
                    w = int(plate[2] - plate[0]) # xmax - xmin
                    h = int(plate[3] - plate[1]) # ymax - ymin
                    crop_img = img[y:y+h, x:x+w]
                    print("crop_img", y+h, x+w)
                    cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
                    # cv2.imwrite("crop.jpg", crop_img)
                    # rc_image = cv2.imread("crop.jpg")
                    lp = ""
                    for cc in range(0,2):
                        for ct in range(0,2):
                            lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                            print("BSX:", lp)
                            if lp != "unknown":
                                list_read_plates.append(lp)
                                cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                                flag = 1
                                break
                        if flag == 1:
                            break
            print("lpll", list_read_plates)
            if list_read_plates:
                print("lpll",list_read_plates)
                pathDestination = os.path.join('dataset_output', 'license_plate_recognized', filepath[25:])
                pathDestination = pathDestination.split("\\")
                if not os.path.exists(os.path.join(pathDestination[0], pathDestination[1], pathDestination[2],
                                                   pathDestination[3])):
                    print("Create New Folder")
                    os.makedirs(os.path.join(pathDestination[0], pathDestination[1], pathDestination[2],
                                             pathDestination[3]))
                list_read_plates = '_'.join(map(str, list_read_plates))
                pathDestination1 = list_read_plates.replace("-", "_") + "_" + pathDestination[4]
                pathDestination = os.path.join(pathDestination[0], pathDestination[1], pathDestination[2], pathDestination[3], pathDestination1)
                print("Đete",pathDestination)
                shutil.move(filepath.replace("\\","/"), pathDestination.replace("\\","/"))
            else:
                print("unkn")
                pathDestination = os.path.join('dataset_output', 'unknown', filepath[25:])
                pathDestination = pathDestination.split("\\")
                if not os.path.exists(os.path.join(pathDestination[0], pathDestination[1], pathDestination[2], pathDestination[3])):
                    print("Create New Folder")
                    os.makedirs(os.path.join(pathDestination[0], pathDestination[1], pathDestination[2], pathDestination[3]))

                pathDestination = os.path.join(pathDestination[0], pathDestination[1], pathDestination[2], pathDestination[3], pathDestination[4])
                print("nodete",pathDestination)
                shutil.move(filepath.replace("\\","/"), pathDestination.replace("\\","/"))


    # if i1 == line:
    #     print("end line")
    #     # break
# cv2.imshow('frame', img)
# cv2.waitKey()
# cv2.destroyAllWindows()