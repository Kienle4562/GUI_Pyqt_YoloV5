import string
import sys
from random import randint

from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget, QFileDialog,
)
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap, QPainter

# global path
path = 'images\ImgDrawCoordinates.jpg'
PointA = []
PointB = []
def Ch():
    global path
    fileName, fileType = QFileDialog.getOpenFileName(None,'Choose file', '', '*.jpg *.png *.tif *.jpeg')
    if fileName:
        path = fileName
        print("filenamr", fileName)
    return fileName
class AnotherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.window_width, self.window_height = 1200, 800
        self.setMinimumSize(self.window_width, self.window_height)

        layout = QVBoxLayout()
        self.setLayout(layout)
        # fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        # if fileName:
        #     path = fileName
        #     # print(path)
        # print("filer",path)
        self.pix = QPixmap(path)
        # self.pix.fill(Qt.white)
        self.begin, self.destination = QPoint(), QPoint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(QPoint(), self.pix)
        if not self.begin.isNull() and not self.destination.isNull():
            rect = QRect(self.begin, self.destination)
            painter.drawRect(rect.normalized())

    def mousePressEvent(self, event):
        global PointA
        if event.buttons() & Qt.LeftButton:
            print('Point 1')
            self.begin = event.pos()
            self.destination = self.begin
            self.update()
            PointA = event.pos()
            print("1", event.pos())

    def mouseMoveEvent(self, event):
        global PointB
        if event.buttons() & Qt.LeftButton:
            print('Point 2')
            self.destination = event.pos()
            self.update()
            PointB = str(event.pos())[str(event.pos()).find('(')+1: str(event.pos()).find(')')].split(",")
            print("PointB", PointB)
            print("PointB2", PointB[0])



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.window1 = AnotherWindow()
        self.window2 = AnotherWindow()

        l = QVBoxLayout()
        button1 = QPushButton("Push for Window 1")
        button1.clicked.connect(self.toggle_window1)
        l.addWidget(button1)

        button2 = QPushButton("Push for Window 2")
        button2.clicked.connect(self.toggle_window2)
        l.addWidget(button2)

        w = QWidget()
        w.setLayout(l)
        self.setCentralWidget(w)

    def toggle_window1(self, checked):
        # Select image config file to read
        # fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        # if fileName:
        #     path = fileName
        #     # print(path)
        #     print("filenamr",fileName)
        # Ch()
        if self.window1.isVisible():
            self.window1.hide()

        else:
            self.window1.show()

    def toggle_window2(self, checked):
        if self.window2.isVisible():
            self.window2.hide()

        else:
            self.window2.show()


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()