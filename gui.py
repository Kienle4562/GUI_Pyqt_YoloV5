# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1280, 720)
        self.gridLayout_6 = QtWidgets.QGridLayout(Form)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.tabWidget = QtWidgets.QTabWidget(self.groupBox_3)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.horizontalLayout_Y3 = QtWidgets.QHBoxLayout(self.tab_5)
        self.horizontalLayout_Y3.setObjectName("horizontalLayout_Y3")
        self.formLayout_Y3 = QtWidgets.QFormLayout()
        self.formLayout_Y3.setObjectName("formLayout_Y3")
        self.lb_Y1 = QtWidgets.QLabel(self.tab_5)
        self.lb_Y1.setObjectName("lb_Y1")
        self.formLayout_Y3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lb_Y1)
        self.lineEdit = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit.setEnabled(True)
        self.lineEdit.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout_Y3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.lb_Y1_2 = QtWidgets.QLabel(self.tab_5)
        self.lb_Y1_2.setObjectName("lb_Y1_2")
        self.formLayout_Y3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.lb_Y1_2)
        self.pushButton_upload_yolo = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_upload_yolo.setObjectName("pushButton_upload_yolo")
        self.formLayout_Y3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.pushButton_upload_yolo)
        self.lb_8 = QtWidgets.QLabel(self.tab_5)
        self.lb_8.setObjectName("lb_8")
        self.formLayout_Y3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.lb_8)
        self.box_7 = QtWidgets.QComboBox(self.tab_5)
        self.box_7.setObjectName("box_7")
        self.box_7.addItem("")
        self.box_7.addItem("")
        self.formLayout_Y3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.box_7)
        self.lb_2 = QtWidgets.QLabel(self.tab_5)
        self.lb_2.setObjectName("lb_2")
        self.formLayout_Y3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.lb_2)
        self.box_8 = QtWidgets.QComboBox(self.tab_5)
        self.box_8.setObjectName("box_8")
        self.box_8.addItem("")
        self.box_8.addItem("")
        self.box_8.addItem("")
        self.box_8.addItem("")
        self.formLayout_Y3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.box_8)
        self.lb_7 = QtWidgets.QLabel(self.tab_5)
        self.lb_7.setObjectName("lb_7")
        self.formLayout_Y3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.lb_7)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_4.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.formLayout_Y3.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_4)
        self.lb_3 = QtWidgets.QLabel(self.tab_5)
        self.lb_3.setObjectName("lb_3")
        self.formLayout_Y3.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.lb_3)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout_Y3.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.buttonUpdateSettings = QtWidgets.QPushButton(self.tab_5)
        self.buttonUpdateSettings.setObjectName("buttonUpdateSettings")
        self.formLayout_Y3.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.buttonUpdateSettings)
        self.horizontalLayout_Y3.addLayout(self.formLayout_Y3)
        self.tabWidget.addTab(self.tab_5, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.tab)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.lb_1 = QtWidgets.QLabel(self.tab)
        self.lb_1.setObjectName("lb_1")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lb_1)
        self.box_1 = QtWidgets.QPushButton(self.tab)
        self.box_1.setAutoRepeatInterval(100)
        self.box_1.setDefault(True)
        self.box_1.setObjectName("box_1")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.box_1)
        self.lb_21 = QtWidgets.QLabel(self.tab)
        self.lb_21.setObjectName("lb_21")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.lb_21)
        self.box_2 = QtWidgets.QComboBox(self.tab)
        self.box_2.setObjectName("box_2")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.box_2)
        self.lb_31 = QtWidgets.QLabel(self.tab)
        self.lb_31.setObjectName("lb_31")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.lb_31)
        self.box_3 = QtWidgets.QComboBox(self.tab)
        self.box_3.setObjectName("box_3")
        self.box_3.addItem("")
        self.box_3.addItem("")
        self.box_3.addItem("")
        self.box_3.addItem("")
        self.box_3.addItem("")
        self.box_3.addItem("")
        self.box_3.addItem("")
        self.box_3.addItem("")
        self.box_3.addItem("")
        self.box_3.addItem("")
        self.box_3.addItem("")
        self.box_3.addItem("")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.box_3)
        self.horizontalLayout_3.addLayout(self.formLayout_3)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.formLayout_5 = QtWidgets.QFormLayout()
        self.formLayout_5.setObjectName("formLayout_5")
        self.lb_4 = QtWidgets.QLabel(self.tab_2)
        self.lb_4.setObjectName("lb_4")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lb_4)
        self.box_5 = QtWidgets.QComboBox(self.tab_2)
        self.box_5.setObjectName("box_5")
        self.box_5.addItem("")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.box_5)
        self.lb_5 = QtWidgets.QLabel(self.tab_2)
        self.lb_5.setObjectName("lb_5")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.lb_5)
        self.box_4 = QtWidgets.QComboBox(self.tab_2)
        self.box_4.setObjectName("box_4")
        self.box_4.addItem("")
        self.box_4.addItem("")
        self.box_4.addItem("")
        self.box_4.addItem("")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.box_4)
        self.lb_6 = QtWidgets.QLabel(self.tab_2)
        self.lb_6.setObjectName("lb_6")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.lb_6)
        self.box_6 = QtWidgets.QComboBox(self.tab_2)
        self.box_6.setObjectName("box_6")
        self.box_6.addItem("")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.box_6)
        self.verticalLayout_14.addLayout(self.formLayout_5)
        self.tabWidget.addTab(self.tab_2, "")

        self.horizontalLayout_9.addWidget(self.tabWidget)
        self.verticalLayout_3.addWidget(self.groupBox_3)
        self.state_label = QtWidgets.QLabel(Form)
        self.state_label.setMinimumSize(QtCore.QSize(0, 30))
        self.state_label.setMaximumSize(QtCore.QSize(180, 16777215))
        self.state_label.setTextFormat(QtCore.Qt.AutoText)
        self.state_label.setScaledContents(True)
        self.state_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.state_label.setObjectName("state_label")
        self.verticalLayout_3.addWidget(self.state_label)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem)
        self.checkBox_circle = QtWidgets.QCheckBox(Form)
        self.checkBox_circle.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.checkBox_circle.setObjectName("checkBox_circle")
        self.horizontalLayout_7.addWidget(self.checkBox_circle)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.open_button = QtWidgets.QPushButton(Form)
        self.open_button.setObjectName("open_button")
        self.verticalLayout_3.addWidget(self.open_button)
        self.formGroupBox = QtWidgets.QGroupBox(Form)
        self.formGroupBox.setObjectName("formGroupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.formGroupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.send_lcdNumber = QtWidgets.QLCDNumber(self.formGroupBox)
        self.send_lcdNumber.setObjectName("send_lcdNumber")
        self.gridLayout_2.addWidget(self.send_lcdNumber, 0, 1, 1, 1)
        self.rec_lcdNumber = QtWidgets.QLCDNumber(self.formGroupBox)
        self.rec_lcdNumber.setObjectName("rec_lcdNumber")
        self.gridLayout_2.addWidget(self.rec_lcdNumber, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.formGroupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.formGroupBox)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.formGroupBox)
        self.close_button = QtWidgets.QPushButton(Form)
        self.close_button.setObjectName("close_button")
        self.verticalLayout_3.addWidget(self.close_button)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.verticalGroupBox = QtWidgets.QGroupBox(Form)
        self.verticalGroupBox.setObjectName("verticalGroupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalGroupBox)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.receive_text = QtWidgets.QTextBrowser(self.verticalGroupBox)
        self.receive_text.setMaximumSize(QtCore.QSize(16777215, 500))
        self.receive_text.setObjectName("receive_text")
        self.verticalLayout.addWidget(self.receive_text)
        self.verticalLayout_9.addWidget(self.verticalGroupBox)
        self.verticalGroupBox_2 = QtWidgets.QGroupBox(Form)
        self.verticalGroupBox_2.setObjectName("verticalGroupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalGroupBox_2)
        self.verticalLayout_2.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.send_text = QtWidgets.QTextEdit(self.verticalGroupBox_2)
        self.send_text.setMaximumSize(QtCore.QSize(16777215, 500))
        self.send_text.setObjectName("send_text")
        self.verticalLayout_2.addWidget(self.send_text)
        self.verticalLayout_9.addWidget(self.verticalGroupBox_2)
        self.horizontalLayout.addLayout(self.verticalLayout_9)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.hex_receive = QtWidgets.QCheckBox(Form)
        self.hex_receive.setObjectName("hex_receive")
        self.verticalLayout_7.addWidget(self.hex_receive)
        self.clear_button = QtWidgets.QPushButton(Form)
        self.clear_button.setObjectName("clear_button")
        self.verticalLayout_7.addWidget(self.clear_button)
        self.verticalLayout_8.addLayout(self.verticalLayout_7)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.hex_send = QtWidgets.QCheckBox(Form)
        self.hex_send.setObjectName("hex_send")
        self.verticalLayout_6.addWidget(self.hex_send)
        self.clear_button_2 = QtWidgets.QPushButton(Form)
        self.clear_button_2.setObjectName("clear_button_2")
        self.verticalLayout_6.addWidget(self.clear_button_2)
        self.verticalLayout_8.addLayout(self.verticalLayout_6)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.timer_send_cb = QtWidgets.QCheckBox(Form)
        self.timer_send_cb.setObjectName("timer_send_cb")
        self.verticalLayout_5.addWidget(self.timer_send_cb)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.lineEdit_3 = QtWidgets.QLineEdit(Form)
        self.lineEdit_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_5.addWidget(self.lineEdit_3)
        self.dw = QtWidgets.QLabel(Form)
        self.dw.setObjectName("dw")
        self.horizontalLayout_5.addWidget(self.dw)
        self.verticalLayout_5.addLayout(self.horizontalLayout_5)
        self.verticalLayout_8.addLayout(self.verticalLayout_5)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_4.addWidget(self.pushButton)
        self.send_button = QtWidgets.QPushButton(Form)
        self.send_button.setObjectName("send_button")
        self.verticalLayout_4.addWidget(self.send_button)
        self.verticalLayout_8.addLayout(self.verticalLayout_4)
        self.horizontalLayout.addLayout(self.verticalLayout_8)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.gridLayout_6.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.line_2 = QtWidgets.QFrame(Form)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_6.addWidget(self.line_2, 0, 1, 1, 1)
        self.tabWidget_2 = QtWidgets.QTabWidget(Form)
        self.tabWidget_2.setMinimumSize(QtCore.QSize(600, 0))
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.label_video = QtWidgets.QLabel(self.tab_3)
        self.label_video.setMinimumSize(QtCore.QSize(480, 246))
        self.label_video.setMaximumSize(QtCore.QSize(166770, 166770))
        self.label_video.setText("")
        self.label_video.setObjectName("label_video")
        self.verticalLayout_11.addWidget(self.label_video)
        self.line = QtWidgets.QFrame(self.tab_3)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_11.addWidget(self.line)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.textBrowser_video = QtWidgets.QLabel(self.groupBox_2)
        self.textBrowser_video.setObjectName("textBrowser_video")
        self.gridLayout_5.addWidget(self.textBrowser_video, 0, 0, 1, 1)
        self.horizontalLayout_2.addWidget(self.groupBox_2)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.pushButton_streaming = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_streaming.setObjectName("pushButton_streaming")
        self.verticalLayout_12.addWidget(self.pushButton_streaming)
        self.pushButton_loadmp4 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_loadmp4.setObjectName("pushButton_loadmp4")
        self.verticalLayout_12.addWidget(self.pushButton_loadmp4)
        self.pushButton_stopscan = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_stopscan.setObjectName("pushButton_stopscan")
        self.verticalLayout_12.addWidget(self.pushButton_stopscan)
        self.horizontalLayout_2.addLayout(self.verticalLayout_12)
        self.verticalLayout_11.addLayout(self.horizontalLayout_2)
        self.tabWidget_2.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout.setObjectName("gridLayout")
        self.left_img = QtWidgets.QLabel(self.tab_4)
        self.left_img.setMinimumSize(QtCore.QSize(200, 200))
        self.left_img.setMaximumSize(QtCore.QSize(166770, 166770))
        self.left_img.setText("")
        self.left_img.setObjectName("left_img")
        self.gridLayout.addWidget(self.left_img, 0, 0, 1, 1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.groupBox = QtWidgets.QGroupBox(self.tab_4)
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 500))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.textBrowser_pic = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_pic.setMaximumSize(QtCore.QSize(16777215, 500))
        self.textBrowser_pic.setObjectName("textBrowser_pic")
        self.gridLayout_4.addWidget(self.textBrowser_pic, 0, 0, 1, 1)
        self.horizontalLayout_6.addWidget(self.groupBox)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.pushButton_loadpic = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_loadpic.setObjectName("pushButton_loadpic")
        self.verticalLayout_10.addWidget(self.pushButton_loadpic)
        self.pushButton_scanpic = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_scanpic.setObjectName("pushButton_scanpic")
        self.verticalLayout_10.addWidget(self.pushButton_scanpic)
        self.pushButton_comparePic = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_comparePic.setObjectName("pushButton_comparePic")
        self.verticalLayout_10.addWidget(self.pushButton_comparePic)
        self.horizontalLayout_6.addLayout(self.verticalLayout_10)
        self.gridLayout.addLayout(self.horizontalLayout_6, 3, 0, 1, 1)
        self.right_img = QtWidgets.QLabel(self.tab_4)
        self.right_img.setMinimumSize(QtCore.QSize(200, 200))
        self.right_img.setText("")
        self.right_img.setObjectName("right_img")
        self.gridLayout.addWidget(self.right_img, 1, 0, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.tab_4)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout.addWidget(self.line_3, 2, 0, 1, 1)
        self.tabWidget_2.addTab(self.tab_4, "")
        self.gridLayout_6.addWidget(self.tabWidget_2, 0, 2, 1, 1)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox_3.setTitle(_translate("Form", "Serial Port"))
        self.lb_Y1.setText(_translate("Form", "Select Cam:"))
        self.lb_Y1_2.setText(_translate("Form", "Select Yolo Model:"))
        self.pushButton_upload_yolo.setText(_translate("Form", "Upload Model"))
        self.lb_8.setText(_translate("Form", "Select GPU:"))
        self.box_7.setItemText(0, _translate("Form", "CPU"))
        self.box_7.setItemText(1, _translate("Form", "Graphics Card"))
        self.lb_2.setText(_translate("Form", "Choose Object:"))
        self.box_8.setItemText(0, _translate("Form", "0 People"))
        self.box_8.setItemText(1, _translate("Form", "1 Bicycle"))
        self.box_8.setItemText(2, _translate("Form", "2 Car"))
        self.box_8.setItemText(3, _translate("Form", "3 Motorcycle"))
        self.lb_7.setText(_translate("Form", "Point Right:"))
        self.lb_3.setText(_translate("Form", "Point Left:"))
        self.buttonUpdateSettings.setText(_translate("Form", "Update Settings"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("Form", "Yolo Configuration"))
        self.lb_1.setText(_translate("Form", "Serial Port Detection:"))
        self.box_1.setText(_translate("Form", "Detect Serial Port"))
        self.lb_21.setText(_translate("Form", "Serial Port Selection:"))
        self.lb_31.setText(_translate("Form", "Baud Rate:"))
        self.box_3.setItemText(0, _translate("Form", "115200"))
        self.box_3.setItemText(1, _translate("Form", "2400"))
        self.box_3.setItemText(2, _translate("Form", "4800"))
        self.box_3.setItemText(3, _translate("Form", "9600"))
        self.box_3.setItemText(4, _translate("Form", "14400"))
        self.box_3.setItemText(5, _translate("Form", "19200"))
        self.box_3.setItemText(6, _translate("Form", "38400"))
        self.box_3.setItemText(7, _translate("Form", "57600"))
        self.box_3.setItemText(8, _translate("Form", "76800"))
        self.box_3.setItemText(9, _translate("Form", "12800"))
        self.box_3.setItemText(10, _translate("Form", "230400"))
        self.box_3.setItemText(11, _translate("Form", "460800"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "Basic Configuration"))
        self.lb_4.setText(_translate("Form", "Data Bits:"))
        self.box_5.setItemText(0, _translate("Form", "N"))
        self.lb_5.setText(_translate("Form", "Check Digit:"))
        self.box_4.setItemText(0, _translate("Form", "8"))
        self.box_4.setItemText(1, _translate("Form", "7"))
        self.box_4.setItemText(2, _translate("Form", "6"))
        self.box_4.setItemText(3, _translate("Form", "5"))
        self.lb_6.setText(_translate("Form", "Stop Bit:"))
        self.box_6.setItemText(0, _translate("Form", "1"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "Other Configuration"))
        self.state_label.setText(_translate("Form", "Created by ProtoDrive000"))
        self.checkBox_circle.setText(_translate("Form", "Ellipse Detection"))
        self.open_button.setText(_translate("Form", "Open Serial Port"))
        self.formGroupBox.setTitle(_translate("Form", "Serial Port Status"))
        self.label_2.setText(_translate("Form", "Has Been Sent:"))
        self.label.setText(_translate("Form", "Received:"))
        self.close_button.setText(_translate("Form", "Close Serial Port"))
        self.verticalGroupBox.setTitle(_translate("Form", "Reception Area"))
        self.verticalGroupBox_2.setTitle(_translate("Form", "Sending Area"))
        self.send_text.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'SimSun\'; font-size:9pt;\"><br /></p></body></html>"))
        self.hex_receive.setText(_translate("Form", "Hex Receive"))
        self.clear_button.setText(_translate("Form", "Clear"))
        self.hex_send.setText(_translate("Form", "Hex Send"))
        self.clear_button_2.setText(_translate("Form", "Clear"))
        self.timer_send_cb.setText(_translate("Form", "Timing Send"))
        self.lineEdit_3.setText(_translate("Form", "1000"))
        self.dw.setText(_translate("Form", "ms/s"))
        self.pushButton.setText(_translate("Form", "Random Test"))
        self.send_button.setText(_translate("Form", "Send"))
        self.groupBox_2.setTitle(_translate("Form", "Result Output"))
        self.textBrowser_video.setText(_translate("Form", "   "))
        self.pushButton_streaming.setText(_translate("Form", "Live Video Streaming"))
        self.pushButton_loadmp4.setText(_translate("Form", "Video File"))
        self.pushButton_stopscan.setText(_translate("Form", "Stop Detection"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _translate("Form", "Video Stream"))
        self.groupBox.setTitle(_translate("Form", "Result Output"))
        self.pushButton_loadpic.setText(_translate("Form", "Upload Image"))
        self.pushButton_scanpic.setText(_translate("Form", "Image Detection"))
        self.pushButton_comparePic.setText(_translate("Form", "Compared"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _translate("Form", "Still Image"))

