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

n = 'number'
s = 'string'
c = 'characters'
typeLP1 = [n, n, s, n, c, n, n, n, n]
typeLP2 = [n, n, s, n, c, n, n, n, n, n]

# while True:
with open("Sample_OutPut\Image_data_management.txt", 'r') as f:
    for count, line in enumerate(f):
        pass
    line = count + 1
    print('line', line)
    f = open("Sample_OutPut\Image_data_management.txt")
    list_content = f.readlines()
    for path in range(line):
        type_detected = []
        filepath = list_content[path]
        filepath = filepath.replace("\n", "")
        print("file:",filepath)
        if os.path.exists(filepath):
            img = cv2.imread(filepath)
            plates = yolo_LP_detect(img, size=640)
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
                    x = int(plate[0]) # xmin
                    y = int(plate[1]) # ymin
                    w = int(plate[2] - plate[0]) # xmax - xmin
                    h = int(plate[3] - plate[1]) # ymax - ymin
                    crop_img = img[y:y+h, x:x+w]
                    cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
                    cv2.imwrite("crop.jpg", crop_img)
                    # rc_image = cv2.imread("crop.jpg")
                    lp = ""
                    for cc in range(0,2):
                        for ct in range(0,2):
                            lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                            if lp != "unknown":
                                print('newlp',lp)
                                list_read_plates.append(lp)
                                if not os.path.exists('dataset_output/img_display'):
                                    os.makedirs('dataset_output/img_display') # Create New Folder
                                if not os.path.exists('dataset_output/img_display/' + str(lp) + '.jpg'):
                                    cv2.imwrite('dataset_output/img_display/' + str(lp) + '.jpg', crop_img)
                                    type_detected_lp =[]
                                    # if lp:
                                    for i in range(len(lp)):
                                        if lp[i].isdigit():
                                            type_detected_lp.append('number')
                                        elif lp[i].isalpha():
                                            type_detected_lp.append('string')
                                        elif lp[i].find('-') == 0:
                                            type_detected_lp.append('characters')
                                    print("Type1", type_detected_lp)
                                    if type_detected_lp == typeLP1 or type_detected_lp == typeLP2:
                                        print('save LP')
                                        with open("dataset_output\List_Result_License_Plate_Recognition.txt", "a") as f:
                                            f.write(('dataset_output/img_display/' + str(lp) + '.jpg') + "\n")
                                cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                                flag = 1
                                break
                        if flag == 1:
                            break
            print("lpll1", list_read_plates)
            filepathLP = filepath.split('\\')
            if list_read_plates:
                for i in range(len(list_read_plates[0])):
                    if list_read_plates[0][i].isdigit():
                        type_detected.append('number')
                    elif list_read_plates[0][i].isalpha():
                        type_detected.append('string')
                    elif list_read_plates[0][i].find('-') == 0:
                        type_detected.append('characters')
            print("Type2",type_detected)
            if typeLP1 == type_detected or typeLP2 == type_detected:
                pathDestination = os.path.join('dataset_output', 'license_plate_recognized', filepathLP[3], filepathLP[4], filepathLP[5])
                pathDestination = pathDestination.split("\\")
                if not os.path.exists(os.path.join(pathDestination[0], pathDestination[1], pathDestination[2],
                                                   pathDestination[3])):
                    # print("Create New Folder")
                    os.makedirs(os.path.join(pathDestination[0], pathDestination[1], pathDestination[2],
                                             pathDestination[3]))
                list_read_plates = '_'.join(map(str, list_read_plates))
                pathDestination1 = list_read_plates.replace("-", "_") + "_" + pathDestination[4]
                pathDestination = os.path.join(pathDestination[0], pathDestination[1], pathDestination[2], pathDestination[3], pathDestination1)
                shutil.move(filepath.replace("\\","/"), pathDestination.replace("\\","/"))
                print("Move to Img dis", pathDestination)
            else:
                pathDestination = os.path.join('dataset_output', 'unknown', filepathLP[3], filepathLP[4], filepathLP[5])
                pathDestination = pathDestination.split("\\")
                if not os.path.exists(os.path.join(pathDestination[0], pathDestination[1], pathDestination[2], pathDestination[3])):
                    print("Create New Folder")
                    os.makedirs(os.path.join(pathDestination[0], pathDestination[1], pathDestination[2], pathDestination[3]))

                pathDestination = os.path.join(pathDestination[0], pathDestination[1], pathDestination[2], pathDestination[3], pathDestination[4])
                shutil.move(filepath.replace("\\","/"), pathDestination.replace("\\","/"))
                print("Move to unkn", pathDestination)

# cv2.imshow('frame', img)
# cv2.waitKey()
# cv2.destroyAllWindows()