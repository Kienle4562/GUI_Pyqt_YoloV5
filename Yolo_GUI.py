import sys
from datetime import date

import serial
import serial.tools.list_ports
import time
from typing import List
import random

from PyQt5 import QtWidgets
from gui import Ui_Form
import cv2
import numpy as np
import math
import json
from multiprocessing import Pool
import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


from deep_sort.utils.parser import get_config##K
from deep_sort.deep_sort import DeepSort##K
from openpyxl import load_workbook
count = 0 ##K
data = [] ##K


def count_obj(box, w, h, id):
    # print("Tets",box,w,h,id)
    global count, data
    today = date.today()
    # center_coordinates = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))
    # print("Test1",(box[1] + (box[3] - box[1]) / 2),":",(h-350))
    if int(box[1] + (box[3] - box[1]) / 2) > (h - 350):
        if id not in data:
            count += 1
            data.append(id)
            print('count', count)

            d1 = today.strftime("%d/%m/%Y")
            wb = load_workbook('Test.xlsm', keep_vba=True)
            sh = wb.active
            ws = wb['My sheet']
            countA = sh.max_row
            Day1 = sh.cell(row=countA, column=1).value
            # print("counta2",countA,"countA:",countA,"day1",Day1,"d",d1)
            if Day1 == d1:
                ws.cell(row=countA, column=2).value = count
                wb.save('Test.xlsm')
            else:
                countA += 1
                ws.cell(row=countA, column=1).value = d1
                ws.cell(row=countA, column=2).value = count
                wb.save('Test.xlsm')


class MainWindows(QtWidgets.QWidget, Ui_Form):

    def __init__(self):
        super(MainWindows, self).__init__()
        self.setupUi(self)
        self.init()
        self.setWindowTitle("Yolo Host Computer")
        self.ser = serial.Serial()
        self.port_check()
        self.setWindowIcon(QIcon("images/UI/logo.jpg"))

        # The number of received data and sent data is set to zero
        self.data_num_received = 0
        self.data_num_sended = 0
        self.rec_lcdNumber.display(self.data_num_received)
        self.send_lcdNumber.display(self.data_num_sended)

        # Image reading process
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cpu'
        self.deviceName = 'CPU'
        self.classes = '0'
        self.pointRight = '200'
        self.pointLeft = '200'
        self.model_path = "Model_Yolo/yolov5n.pt"
        # Initialize the video read thread
        self.vid_source = '0'  # The initial setting is the camera
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        # self.model = self.model_load(weights=self.model_path,
        #                              device=self.device)
        # todo A device indicating where the model is loaded
        self.model = self.model_load(weights="Model_Yolo/yolov5n.pt",
                                   device='cpu')

        self.reset_vid()

    '''
    ***Model initialization***
    '''
    @torch.no_grad()
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
            print("Mode!", weights)
        print("Model loading is complete!",weights)
        return model
    def reset_vid(self):
        self.pushButton_streaming.setEnabled(True)
        self.pushButton_loadmp4.setEnabled(True)
        self.label_video.setPixmap(QPixmap("images/UI/yolo.jpg"))

        self.label_video.setScaledContents(True)
        # self.vid_source = '0'
        self.webcam = True
    def init(self):
        # Serial port detection button
        self.box_1.clicked.connect(self.port_check)
        #pf.test(100,0.3)
        # Serial port information display
        self.box_2.currentTextChanged.connect(self.port_imf)

        # Open serial button
        self.open_button.clicked.connect(self.port_open)

        # Close serial button
        self.close_button.clicked.connect(self.port_close)

        # Send data button
        self.send_button.clicked.connect(self.data_send)

        # Send data regularly
        self.timer_send = QTimer()
        self.timer_send.timeout.connect(self.data_send)
        self.timer_send_cb.stateChanged.connect(self.data_send_timer)

        # Timer receives data
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.data_receive)

        # Clear send window
        self.clear_button.clicked.connect(self.send_data_clear)

        # Clear receive window
        self.clear_button.clicked.connect(self.receive_data_clear)

        # Upload image window
        self.pushButton_loadpic.clicked.connect(self.upload_img)

        # Detect picture window
        self.pushButton_scanpic.clicked.connect(self.detect_img)
        # Detect video window
        self.pushButton_streaming.clicked.connect(self.open_cam)
        self.pushButton_loadmp4.clicked.connect(self.open_mp4)
        self.pushButton_stopscan.clicked.connect(self.close_vid)

        self.pushButton_upload_yolo.clicked.connect(self.open_model_yolo)
        self.buttonUpdateSettings.clicked.connect(self.yolo_configuration_settings)
    '''
    ***Upload image***
    '''
    def yolo_configuration_settings(self):
        # self.checkBox_Settings.setEnabled(False)
        number1 = self.lineEdit.text()
        number2 = self.lineEdit_4.text()
        number3 = self.lineEdit_2.text()
        # SelectCamera
        if self.lineEdit.text() != "":
            try:
                number1 = int(number1)
                self.vid_source = str(self.lineEdit.text())
            except Exception:
                QMessageBox.about(self, 'Error', 'Input can only be a number \nEx:0 or 1 or...')
                pass
        else:
            self.vid_source = '0'
        self.lineEdit.setText(str(self.vid_source))
        #Select GPU
        if str(self.box_7.currentText()) == 'Graphics Card':
            self.device = str('0') #GPU
            self.deviceName = str('Graphics Card')
        else:
            self.device = str('cpu') #CPU
            self.deviceName = str('CPU')
        # Select Object
        object = str(self.box_8.currentText())
        self.classes = object[0:1]
        #Point Rigth
        if self.lineEdit_4.text() != "":
            try:
                number2 = int(number2)
                self.pointRight = int(self.lineEdit_4.text())
            except Exception:
                QMessageBox.about(self, 'Error', 'Input point rigth can only be number')
                pass
        else:
            self.pointRight = 250
        self.lineEdit_4.setText(str(self.pointRight))
        #Point Left
        if self.lineEdit_2.text() != "":
            try:
                number3 = int(number3)
                self.pointLeft = int(self.lineEdit_2.text())
            except Exception:
                QMessageBox.about(self, 'Error', 'Input point left can only be a number')
                pass
        else:
            self.pointLeft = 250
        self.lineEdit_2.setText(str(self.pointLeft))

        # filepath_not_exist = str(self.model_path)
        basename = os.path.basename(self.model_path)
        self.pushButton_upload_yolo.setText(str(basename))
        # Load model #Select Yolo Model
        fileName = self.model_path
        if fileName:
            self.model = self.model_load(weights=str(fileName),
                            device=str(self.device))  #
            print("Upload model yolo complete:",str(fileName))
            self.textBrowser_pic.setText("Upload model yolo complete")
            self.textBrowser_video.setText("Upload model yolo complete")
            self.pushButton_upload_yolo.setText(basename)

        #Show resul
        t= "Cam:"+str(self.vid_source)+"  Yolo Model:"+str(basename)+"  GPU:"+str(self.device)+'\n'+"Object:"+str(object)+"  Point Right:"+str(self.pointRight)+"  Point Left:"+str(self.pointLeft)
        self.textBrowser_video.setText(str(t))
        self.textBrowser_pic.setText(str(t))
        # Load model
        fileName = self.model_path
        if fileName:
            self.model = self.model_load(weights=str(fileName),
                                         device=str(self.device))  #
            print("Upload model yolo complete:", str(fileName))
            self.pushButton_upload_yolo.setText(basename)
        QMessageBox.about(self, 'Complete', 'Configuration has been updated')

    def open_model_yolo(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.pt')
        filepath_not_exist = str(fileName)
        basename = os.path.basename(filepath_not_exist)
        self.model_path = fileName
        if fileName:
            print("Upload model yolo complete:",str(fileName))
            self.textBrowser_pic.setText("Upload model yolo complete")
            self.textBrowser_video.setText("Upload model yolo complete")
            self.pushButton_upload_yolo.setText(basename)

    def upload_img(self):
        # Select the video file to read
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # You should resize the pictures and put them together
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            self.left_img.setScaledContents(True)

            # todo The image on the right is reset after uploading the image，
            self.right_img.setPixmap(QPixmap("images/UI/logo.jpg"))
            self.right_img.setScaledContents(True)
    '''
    ***Detect pictures***
    '''
    def detect_img(self):
        self.lineEdit.setText(str(self.vid_source))
        self.lineEdit_4.setText('0')
        self.lineEdit_2.setText('0')
        self.pushButton_upload_yolo.setText(str(os.path.basename(self.model_path)))

        model = self.model
        print("DV:",self.device)
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = 640  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        # device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = int(self.classes)  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        print(source)
        if source == "":
            QMessageBox.warning(self, "Please Upload", "Please Upload Pictures Before Testing")
        else:
            source = str(source)
            device = select_device(self.device)
            webcam = False
            stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            # Dataloader
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = 1  # batch_size
            vid_path, vid_writer = [None] * bs, [None] * bs
            # Run inference
            if pt and device.type != 'cpu':
                print("Issues warmup")
                # model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
            dt, seen = [0.0, 0.0, 0.0], 0
            for path, im, im0s, vid_cap, s in dataset:
                t1 = time_sync()
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1
                # Inference
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3
                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        count_object = 0
                        for *xyxy, conf, cls in reversed(det):
                            count_object= count_object + 1
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                # with open(txt_path + '.txt', 'a') as f:
                                #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                # str_box = annotator.box_label(xyxy,label, color=colors(c, True))
                                str_box = annotator.box_label(xyxy, color=colors(c, True))

                                # t = str(xyxy)+str(label)
                                t = str(xyxy)+" Object has been counted: "+str(count_object)
                                with open("temp.txt", "w") as f:
                                    f.write(t + "\n")
                                self.textBrowser_pic.setText(t)
                                #if save_crop:
                                #     save_one_box(xyxy, imc, file="P:/Item/Yolov5/yolov5-mask-42-master/images/box" / names[c] / f'{p.stem}.jpg',
                                #                 BGR=True)
                    # Print time (inference-only)
                    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                    # Stream results
                    im0 = annotator.result()
                    # if view_img:
                    #     cv2.imshow(str(p), im0)
                    #     cv2.waitKey(1)  # 1 millisecond
                    # Save results (image with detections)
                    resize_scale = output_size / im0.shape[0]
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("images/tmp/single_result.jpg", im0)
                    # From the current situation, it should only be a problem under ubuntu, but it is complete under windows, so continue
                    if self.checkBox_circle.checkState() > 0:
                        # read input
                        img = cv2.imread('images/tmp/upload_show_result.jpg')

                        # convert to gray
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # threshold
                        thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]

                        # find largest contour
                        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours = contours[0] if len(contours) == 2 else contours[1]
                        if len(contours) != 0:
                            big_contour = max(contours, key=cv2.contourArea)
                            # fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree
                            ellipse = cv2.fitEllipse(big_contour)#big_contour)
                            (xc, yc), (d1, d2), angle = ellipse
                            # print(xc, yc, d1, d1, angle)
                            # print("big_contour",big_contour)
                            # draw ellipse
                            result = img.copy()
                            cv2.ellipse(result, ellipse, (0, 255, 0), 3)

                            # draw circle at center
                            xc, yc = ellipse[0]
                            cv2.circle(result, (int(xc), int(yc)), 10, (255, 255, 255), -1)

                            # draw vertical line
                            # compute major radius
                            rmajor = max(d1, d2) / 2
                            if angle > 90:
                                angle = angle - 90
                            else:
                                angle = angle + 90
                            print(angle)
                            xtop = xc + math.cos(math.radians(angle)) * rmajor
                            ytop = yc + math.sin(math.radians(angle)) * rmajor
                            xbot = xc + math.cos(math.radians(angle + 180)) * rmajor
                            ybot = yc + math.sin(math.radians(angle + 180)) * rmajor
                            cv2.line(result, (int(xtop), int(ytop)), (int(xbot), int(ybot)), (0, 0, 255), 3)
                            cv2.imwrite("images/tmp/single_result.jpg", result)
                        else:
                            cv2.imwrite("images/tmp/single_result.jpg", thresh)
                    self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
                    self.right_img.setScaledContents(True)
    # Video detection, the logic is basically the same, there are two functions, namely, the function of detecting the camera and the function of detecting video files, and the function of detecting the camera first。

    '''
    ### UI close event ### 
    '''
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'Quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    '''
    ### Video off event ### 
    '''

    def open_cam(self):
        self.pushButton_streaming.setEnabled(False)
        self.pushButton_loadmp4.setEnabled(False)
        self.pushButton_stopscan.setEnabled(True)
        # self.vid_source = '0'
        self.webcam = True
        # Reset the button to him
        # print("GOGOGO")
        th = threading.Thread(target=self.detect_vid)
        th.start()

    '''
    ### Enable video file detection event ### 
    '''

    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.pushButton_streaming.setEnabled(False)
            self.pushButton_loadmp4.setEnabled(False)
            # self.pushButton_stopscan.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()


    '''
    ### Video start event ### 
    '''

    # The main functions of the video and the camera are the same, but the incoming source is different.
    def detect_vid(self):
        # pass

        self.buttonUpdateSettings.setEnabled(False)
        self.lineEdit.setText(str(self.vid_source)) #CAM
        self.pushButton_upload_yolo.setText(str(os.path.basename(self.model_path))) #Model Yolo
        self.lineEdit_4.setText(str(self.pointRight)) #Point Right
        self.lineEdit_2.setText(str(self.pointLeft)) #Point Left



        model = self.model
        print("MD",self.model_path)
        output_size = self.output_size
        # source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = 640  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        # device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = int(self.classes) #None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        source = str(self.vid_source)
        webcam = self.webcam
        device = select_device(self.device)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_img = not nosave and not source.endswith('.txt')  # save inference images

        config_deepsort = 'deep_sort/configs/deep_sort.yaml'##K
        deep_sort_model = 'osnet_x0_25'##

        project = "runs/track"
        name = 'exp'
        exist_ok = False
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        txt_file_name = source.split('/')[-1].split('.')[0]
        txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        cfg = get_config()##K
        cfg.merge_from_file(config_deepsort)
        deepsort = DeepSort(deep_sort_model,
                            max_dist=cfg.DEEPSORT.MAX_DIST,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)##K

        device = select_device(device)
        half &= device.type != 'cpu'

        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()

        if pt and device.type != 'cpu':
            print("Issues warmup")
            # model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # im.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + (
                #     '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop

                cv2.imwrite("images/tmp/single_org_vid.jpg", im0)
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                w, h = im0.shape[1], im0.shape[0]##K
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    xywhs = xyxy2xywh(det[:, 0:4]) ##K
                    confs = det[:, 4]##K
                    clss = det[:, 5]##K

                    t4 = time_sync() ##K
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)##K
                    t5 = time_sync()##K
                    dt[3] += t5 - t4##K

                    # Implement only det_max with the highest confidence in all annotation boxes
                    MaxConf = []
                    len_det = len(det)

                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            #count
                            count_obj(bboxes,w,h,id)
                            c = int(cls)  # integer class
                            # TODO show text 'car 0.....'
                            # label = f'{id} {names[c]} {conf:.2f}'
                            # annotator.box_label(bboxes, label, color=colors(c, True))

                            annotator.box_label(bboxes, color=colors(c, True))

                            # if save_txt:
                            #     # to MOT format
                            #     bbox_left = output[0]
                            #     bbox_top = output[1]
                            #     bbox_w = output[2] - output[0]
                            #     bbox_h = output[3] - output[1]
                            #     # Write MOT compliant results to file
                            #     with open(txt_path, 'a') as f:
                            #         f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                            #                                        bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                    for i in range(len_det):
                        # print(i)
                        MaxConf.append(det[i][4])
                        Max = max(MaxConf)
                        # maximum value
                        MaxI = MaxConf.index(Max)
                    det_max = det[[MaxI]]
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            # str_box = annotator.box_label(xyxy, label, color=colors(c, True))
                            # t = str(xyxy)
                            # print("604", str(label), str(xyxy));
                            # with open("temp.txt", "w") as f:
                            #     f.write(t+"\n")
                            #
                            # self.textBrowser_video.setText(str(count)) ###Loi in hear

                            # if save_crop:
                            #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                            #                  BGR=True)
                #self.label_video.setText(str_box)
                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                # Stream results
                # Save results (image with detections)

                im0 = annotator.result()
                frame = im0
                resize_scale = output_size / frame.shape[0]
                frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)

                org = (150, 150)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 3
                color = (0, 255, 0)
                thickness = 3
                frame_resized = cv2.putText(im0, str(count), org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

                color = (0, 255, 0)
                start_point = (0, h - int(self.pointRight))
                end_point = (w, h - int(self.pointLeft))
                frame_resized = cv2.line(im0, start_point, end_point, color, thickness=1)

                cv2.imwrite("images/tmp/single_result_vid.jpg", frame_resized)
                if self.checkBox_circle.checkState() > 0:
                    # read input
                    img = cv2.imread('images/tmp/single_org_vid.jpg')

                    # convert to gray
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # threshold
                    thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]

                    # find largest contour
                    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]
                    if len(contours) != 0:
                        big_contour = max(contours, key=cv2.contourArea)
                        big_contour
                        # fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree
                        ellipse = cv2.fitEllipse(big_contour)
                        (xc, yc), (d1, d2), angle = ellipse
                        # print(xc, yc, d1, d1, angle)

                        # draw ellipse
                        result = img.copy()

                        cv2.ellipse(result, ellipse, (0, 255, 0), 3)

                        # draw circle at center
                        xc, yc = ellipse[0]
                        cv2.circle(result, (int(xc), int(yc)), 10, (255, 255, 255), -1)
                        # draw vertical line
                        # compute major radius
                        rmajor = max(d1, d2) / 2
                        if angle > 90:
                            angle = angle - 90
                        else:
                            angle = angle + 90
                        # print(angle)
                        xtop = xc + math.cos(math.radians(angle)) * rmajor
                        ytop = yc + math.sin(math.radians(angle)) * rmajor
                        xbot = xc + math.cos(math.radians(angle + 180)) * rmajor
                        ybot = yc + math.sin(math.radians(angle + 180)) * rmajor
                        cv2.line(result, (int(xtop), int(ytop)), (int(xbot), int(ybot)), (0, 0, 255), 3)
                        cv2.imwrite("images/tmp/single_result_vid.jpg", result)
                    else:
                        cv2.imwrite("images/tmp/single_result_vid.jpg", thresh)
                #self.label_video.setPixmap(QPixmap("test.jpg"))
                self.label_video.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                self.label_video.setScaledContents(True)

                # self.label_video
                # if view_img:
                # cv2.imshow(str(p), im0)
                # self.label_video.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                # cv2.waitKey(1)  # 1 millisecond

            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.pushButton_streaming.setEnabled(True)
                self.pushButton_loadmp4.setEnabled(True)
                self.reset_vid()
                break
        # self.reset_vid()

    '''
    ### Video reset event ### 
    '''

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()
        self.buttonUpdateSettings.setEnabled(True)

    # Serial port detection
    def port_check(self):
        # Detect all existing serial ports, store the information in a dictionary
        self.Com_Dict = {}
        port_list = list(serial.tools.list_ports.comports())
        self.box_2.clear()
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
            self.box_2.addItem(port[0])
        if len(self.Com_Dict) == 0:
            self.state_label.setText(" No Serial Port")

    # Serial port information
    def port_imf(self):
        # Display details of the selected serial port
        imf_s = self.box_2.currentText()
        if imf_s != "":
            self.state_label.setText(self.Com_Dict[self.box_2.currentText()])

    # Open serial port
    def port_open(self):
        self.ser.port = self.box_2.currentText()
        self.ser.baudrate = int(self.box_3.currentText())
        self.ser.bytesize = int(self.box_4.currentText())
        self.ser.stopbits = int(self.box_6.currentText())
        self.ser.parity = self.box_5.currentText()

        try:
            self.ser.open()
        except:
            QMessageBox.critical(self, "Port Error", "This Serial Port Cannot Be Opened！")
            return None

        # Open the serial port receiving timer, the period is 2ms
        self.timer.start(2)

        if self.ser.isOpen():
            self.open_button.setEnabled(False)
            self.close_button.setEnabled(True)
            self.formGroupBox.setTitle("Serial Port Status (Opened) ")

    # Close serial port
    def port_close(self):
        self.timer.stop()
        self.timer_send.stop()
        try:
            self.ser.close()
        except:
            pass
        self.open_button.setEnabled(True)
        self.close_button.setEnabled(False)
        self.lineEdit_3.setEnabled(True)
        # The number of received data and sent data is set to zero
        self.data_num_received = 0
        #self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        #self.lineEdit_2.setText(str(self.data_num_sended))
        self.rec_lcdNumber.display(self.data_num_received)
        self.send_lcdNumber.display(self.data_num_sended)
        self.formGroupBox.setTitle("Serial Port Status (Closed) ")

    # Send data
    def data_send(self):
        if self.ser.isOpen():
            input_s = self.send_text.toPlainText()
            if input_s != "":
                # Non-empty string
                if self.hex_send.isChecked():
                    # Hex send
                    input_s = input_s.strip()
                    send_list = []
                    while input_s != '':
                        try:
                            num = int(input_s[0:2], 16)
                        except ValueError:
                            QMessageBox.critical(self, 'wrong data', 'Please Enter Hexadecimal Data, Separated By Spaces!')
                            return None
                        input_s = input_s[2:].strip()
                        send_list.append(num)
                    input_s = bytes(send_list)
                else:
                    # Ascii send
                    input_s = (input_s + '\r\n').encode('utf-8')

                num = self.ser.write(input_s)
                self.data_num_sended += num
                #self.lineEdit_2.setText(str(self.data_num_sended))
                self.send_lcdNumber.display(self.data_num_sended)
        else:
            pass

    # Receive Data
    def data_receive(self):
        try:
            num = self.ser.inWaiting()
        except:
            self.port_close()
            return None
        if num > 0:
            data = self.ser.read(num)
            num = len(data)
            # Hex display
            if self.hex_receive.checkState():
                out_s = ''
                for i in range(0, len(data)):
                    out_s = out_s + '{:02X}'.format(data[i]) + ' '
                self.receive_text.insertPlainText(out_s)
            else:
                # The string received by the serial port is b'123', which needs to be converted into a unicode string before it can be output to the window.
                self.receive_text.insertPlainText(data.decode('iso-8859-1'))

            # Count the number of received characters
            self.data_num_received += num
            #self.lineEdit.setText(str(self.data_num_received))
            self.rec_lcdNumber.display(self.data_num_received)
            # Get the text cursor
            textCursor = self.receive_text.textCursor()
            # Scroll to bottom
            textCursor.movePosition(textCursor.End)
            # Set the cursor to the text
            self.receive_text.setTextCursor(textCursor)
        else:
            pass

    # Send data regularly
    def data_send_timer(self):
        if self.timer_send_cb.isChecked():
            self.timer_send.start(int(self.lineEdit_3.text()))
            self.lineEdit_3.setEnabled(False)
        else:
            self.timer_send.stop()
            self.lineEdit_3.setEnabled(True)

    # Clear display
    def send_data_clear(self):
        self.send_text.setText("")

    def receive_data_clear(self):
        self.receive_text.setText("")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myshow = MainWindows()
    myshow.show()
    sys.exit(app.exec_())
