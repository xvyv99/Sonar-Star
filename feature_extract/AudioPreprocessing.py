# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AudioPreprocessing.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import base64

import cv2
import requests
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QIcon
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QMessageBox
from PyQt5.QtWidgets import QMainWindow


class AudioPreprocessing(QMainWindow):
    def __init__(self):
        super(AudioPreprocessing, self).__init__()
        self.setupUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setWindowIcon(QIcon('./img/IntelligentVoice.ico'))
        MainWindow.setStyleSheet("QScrollArea{\n"
                                 "    background-color:#f0f3fa;\n"
                                 "}\n"
                                 "\n"
                                 "QLabel#label_5{\n"
                                 "    font-family:微软雅黑;\n"
                                 "    color:#7468be;\n"
                                 "    font-weight:bold;\n"
                                 "    font-size:32px;\n"
                                 "}\n"
                                 "QLabel#label_6{\n"
                                 "    font-family:微软雅黑;\n"
                                 "    font-weight:bold;\n"
                                 "}\n"
                                 "QLabel#label_7{\n"
                                 "    font-family:微软雅黑;\n"
                                 "    font-weight:bold;\n"
                                 "}\n"
                                 "QPushButton#UploadAudioFileButton_2{\n"
                                 "    font-family:微软雅黑;\n"
                                 "    color:#7468be;\n"
                                 "    font-weight:bold;\n"
                                 "    font-size:12px;\n"
                                 "    background-color:#b39cd0;\n"
                                 "    border-radius: 8px; \n"
                                 "}\n"
                                 "QPushButton#AudioPreprocessingButton_2{\n"
                                 "    font-family:微软雅黑;\n"
                                 "    color:#000000;\n"
                                 "    font-weight:bold;\n"
                                 "    font-size:12px;\n"
                                 "    background-color:#b39cd0;\n"
                                 "    border-radius: 8px; \n"
                                 "}\n"
                                 "QPushButton#AudioWaveformDisplayButton_2{\n"
                                 "    font-family:微软雅黑;\n"
                                 "    color:#7468be;\n"
                                 "    font-weight:bold;\n"
                                 "    font-size:12px;\n"
                                 "    background-color:#b39cd0;\n"
                                 "    border-radius: 8px; \n"
                                 "}\n"
                                 "QPushButton#FeatherExtractionButton_2{\n"
                                 "    font-family:微软雅黑;\n"
                                 "    color:#7468be;\n"
                                 "    font-weight:bold;\n"
                                 "    font-size:12px;\n"
                                 "    background-color:#b39cd0;\n"
                                 "    border-radius: 8px; \n"
                                 "}\n"
                                 "QLabel#label_4{\n"
                                 "    font-family:微软雅黑;\n"
                                 "    font-weight:bold;\n"
                                 "    font-size:26px;\n"
                                 "    \n"
                                 "}\n"
                                 "QLabel#label{\n"
                                 "    font-weight:bold;\n"
                                 "    font-family:微软雅黑;\n"
                                 "    font-size:18px;\n"
                                 "}\n"
                                 "QLabel#label_2{\n"
                                 "    font-weight:bold;\n"
                                 "    font-family:微软雅黑;\n"
                                 "    font-size:26px;\n"
                                 "}\n"
                                 "QLabel#label_3{\n"
                                 "    font-weight:bold;\n"
                                 "    font-family:微软雅黑;\n"
                                 "    font-size:18px;\n"
                                 "}\n"
                                 "QLabel#label_12{\n"
                                 "    font-weight:bold;\n"
                                 "    font-family:微软雅黑;\n"
                                 "    font-size:18px;\n"
                                 "}\n"
                                 "QLabel#label_8{\n"
                                 "    font-weight:bold;\n"
                                 "    font-family:微软雅黑;\n"
                                 "    font-size:18px;\n"
                                 "}\n"
                                 "QLabel#label_11{\n"
                                 "    font-weight:bold;\n"
                                 "    font-family:微软雅黑;\n"
                                 "    font-size:18px;\n"
                                 "}\n"
                                 "\n"
                                 "QPushButton#SelectWaveLocationButton{\n"
                                 "    font-family:微软雅黑;\n"
                                 "    font-weight:bold;\n"
                                 "    font-size:16px;\n"
                                 "    background-color:#b39cd0;\n"
                                 "    border-radius: 10px; \n"
                                 "}\n"
                                 "QPushButton#OkButton{\n"
                                 "    font-family:微软雅黑;\n"
                                 "    font-weight:bold;\n"
                                 "    font-size:18px;\n"
                                 "    background-color:#b39cd0;\n"
                                 "    border-radius: 25px; \n"
                                 "}\n"
                                 "\n"
                                 "QPushButton#CancelButton{\n"
                                 "    font-family:微软雅黑;\n"
                                 "    font-weight:bold;\n"
                                 "    font-size:18px;\n"
                                 "    background-color:#b39cd0;\n"
                                 "    border-radius: 20px; \n"
                                 "}\n"
                                 "\n"
                                 "QPushButton#UploadAudioFileButton_2:hover\n"
                                 "{\n"
                                 "     color: #000000;\n"
                                 " }\n"
                                 "QPushButton#AudioWaveformDisplayButton_2:hover\n"
                                 "{\n"
                                 "     color: #000000;\n"
                                 " }\n"
                                 "QPushButton#FeatherExtractionButton_2:hover\n"
                                 "{\n"
                                 "     color: #000000;\n"
                                 " }")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(160, 210, 241, 241))
        self.graphicsView.setObjectName("graphicsView")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(470, 100, 71, 101))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("./img/205设置 (2).png"))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(550, 110, 241, 81))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(550, 200, 201, 151))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(450, 420, 91, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_3.setGeometry(QtCore.QRect(550, 370, 201, 151))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.scrollArea_2 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_2.setGeometry(QtCore.QRect(0, 0, 801, 101))
        self.scrollArea_2.setStyleSheet("background-color: rgb(240, 243, 250);")
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 799, 99))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.label_5 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_5.setGeometry(QtCore.QRect(100, 0, 441, 101))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_6.setGeometry(QtCore.QRect(590, 50, 91, 21))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_7.setGeometry(QtCore.QRect(670, 50, 91, 20))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_9 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_9.setGeometry(QtCore.QRect(30, 10, 61, 81))
        self.label_9.setText("")
        self.label_9.setPixmap(QtGui.QPixmap("./img/列表展示.png"))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_10.setGeometry(QtCore.QRect(760, 40, 31, 41))
        self.label_10.setText("")
        self.label_10.setPixmap(QtGui.QPixmap("./img/点点.png"))
        self.label_10.setObjectName("label_10")
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(0, 0, 141, 601))
        self.scrollArea.setStyleSheet("background-color: rgb(240, 243, 250);")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 139, 599))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.UploadAudioFileButton_2 = QtWidgets.QPushButton(self.scrollAreaWidgetContents_3)
        self.UploadAudioFileButton_2.setGeometry(QtCore.QRect(10, 150, 121, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(True)
        font.setWeight(75)
        self.UploadAudioFileButton_2.setFont(font)
        self.UploadAudioFileButton_2.setStyleSheet("background-color: rgb(179, 156, 208);")
        self.UploadAudioFileButton_2.setObjectName("UploadAudioFileButton_2")
        self.AudioPreprocessingButton_2 = QtWidgets.QPushButton(self.scrollAreaWidgetContents_3)
        self.AudioPreprocessingButton_2.setGeometry(QtCore.QRect(10, 310, 121, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(True)
        font.setWeight(75)
        self.AudioPreprocessingButton_2.setFont(font)
        self.AudioPreprocessingButton_2.setStyleSheet("background-color: rgb(179, 156, 208);")
        self.AudioPreprocessingButton_2.setObjectName("AudioPreprocessingButton_2")
        self.AudioWaveformDisplayButton_2 = QtWidgets.QPushButton(self.scrollAreaWidgetContents_3)
        self.AudioWaveformDisplayButton_2.setGeometry(QtCore.QRect(10, 390, 121, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(True)
        font.setWeight(75)
        self.AudioWaveformDisplayButton_2.setFont(font)
        self.AudioWaveformDisplayButton_2.setStyleSheet("background-color: rgb(179, 156, 208);")
        self.AudioWaveformDisplayButton_2.setObjectName("AudioWaveformDisplayButton_2")
        self.FeatherExtractionButton_2 = QtWidgets.QPushButton(self.scrollAreaWidgetContents_3)
        self.FeatherExtractionButton_2.setGeometry(QtCore.QRect(10, 230, 121, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(True)
        font.setWeight(75)
        self.FeatherExtractionButton_2.setFont(font)
        self.FeatherExtractionButton_2.setStyleSheet("background-color: rgb(179, 156, 208);")
        self.FeatherExtractionButton_2.setObjectName("FeatherExtractionButton_2")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_3)
        self.scrollArea_3 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea_3.setGeometry(QtCore.QRect(140, 100, 661, 501))
        self.scrollArea_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollArea_3.setObjectName("scrollArea_3")
        self.scrollAreaWidgetContents_4 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_4.setGeometry(QtCore.QRect(0, 0, 659, 499))
        self.scrollAreaWidgetContents_4.setObjectName("scrollAreaWidgetContents_4")
        self.label_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents_4)
        self.label_2.setGeometry(QtCore.QRect(120, 20, 101, 61))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents_4)
        self.label.setGeometry(QtCore.QRect(60, 20, 51, 51))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("./img/205设置 (3).png"))
        self.label.setObjectName("label")
        self.label_12 = QtWidgets.QLabel(self.scrollAreaWidgetContents_4)
        self.label_12.setGeometry(QtCore.QRect(80, 360, 111, 41))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.OkButton = QtWidgets.QPushButton(self.scrollAreaWidgetContents_4)
        self.OkButton.setGeometry(QtCore.QRect(230, 430, 111, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.OkButton.setFont(font)
        self.OkButton.setStyleSheet("background-color: rgb(179, 156, 208);")
        self.OkButton.setObjectName("OkButton")
        self.label_8 = QtWidgets.QLabel(self.scrollAreaWidgetContents_4)
        self.label_8.setGeometry(QtCore.QRect(330, 160, 54, 41))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_2.raise_()
        self.label.raise_()
        self.OkButton.raise_()
        self.label_12.raise_()
        self.label_8.raise_()
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_4)
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(190, 170, 238, 361))
        self.label_13.setText("")
        self.label_13.setPixmap(QtGui.QPixmap("./img/竖线500.png"))
        self.label_13.setObjectName("label_13")
        self.scrollArea_3.raise_()
        self.scrollArea.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.graphicsView_2.raise_()
        self.label_11.raise_()
        self.graphicsView_3.raise_()
        self.scrollArea_2.raise_()
        self.graphicsView.raise_()
        self.label_13.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.OkButton.clicked.connect(self.OkButton_clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def OkButton_clicked(self):
        try:
            r = self.send_requests()

            if r.json().get("return_msg") == "success":
                # 原始波形
                img = r.json().get("originWave")
                img_decode_ = img.encode('ascii')  # ascii编码
                img_decode = base64.b64decode(img_decode_)  # base64解码
                img_np = np.frombuffer(img_decode, np.uint8)  # 从byte数据读取为np.array形式
                img = cv2.imdecode(img_np, cv2.COLOR_RGB2BGR)  # 转为OpenCV形式
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.35)
                x = img.shape[1]
                y = img.shape[0]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame = QImage(img.data, x, y, x * 3, QtGui.QImage.Format_RGB888)
                self.item = QGraphicsPixmapItem(QPixmap.fromImage(frame))
                self.scene = QGraphicsScene()
                self.scene.addItem(self.item)
                self.graphicsView.setScene(self.scene)
                self.graphicsView.show()
                # 降噪波形
                img = r.json().get("reduceWave")
                img_decode_ = img.encode('ascii')  # ascii编码
                img_decode = base64.b64decode(img_decode_)  # base64解码
                img_np = np.frombuffer(img_decode, np.uint8)  # 从byte数据读取为np.array形式
                img = cv2.imdecode(img_np, cv2.COLOR_RGB2BGR)  # 转为OpenCV形式
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.35)
                x = img.shape[1]
                y = img.shape[0]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame = QImage(img.data, x, y, x * 3, QtGui.QImage.Format_RGB888)
                self.item = QGraphicsPixmapItem(QPixmap.fromImage(frame))
                self.scene = QGraphicsScene()
                self.scene.addItem(self.item)
                self.graphicsView_2.setScene(self.scene)
                self.graphicsView_2.show()
                # 端点检测
                img = r.json().get("endpointDetection")
                img_decode_ = img.encode('ascii')  # ascii编码
                img_decode = base64.b64decode(img_decode_)  # base64解码
                img_np = np.frombuffer(img_decode, np.uint8)  # 从byte数据读取为np.array形式
                img = cv2.imdecode(img_np, cv2.COLOR_RGB2BGR)  # 转为OpenCV形式
                img = cv2.resize(img, (0, 0), fx=0.35, fy=0.25)
                x = img.shape[1]
                y = img.shape[0]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame = QImage(img.data, x, y, x * 3, QtGui.QImage.Format_RGB888)
                self.item = QGraphicsPixmapItem(QPixmap.fromImage(frame))
                self.scene = QGraphicsScene()
                self.scene.addItem(self.item)
                self.graphicsView_3.setScene(self.scene)
                self.graphicsView_3.show()
            else:
                QMessageBox.information(self, "提示", "请先进行音频处理", QMessageBox.Yes)
        except Exception as e:
            QMessageBox.information(self, "提示", "请先进行音频处理", QMessageBox.Yes)

    def send_requests(self):
        r = requests.post("http://119.3.239.86:5000/getwave")
        return r

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "自闭症谱系障碍儿童声学特征滤波识别诊断系统"))
        self.label_4.setText(_translate("MainWindow", "处理后音频波形图"))
        self.label_11.setText(_translate("MainWindow", "端点检测："))
        self.label_5.setText(_translate("MainWindow", "声纳星鉴  Intelligent Voice"))
        self.label_6.setText(_translate("MainWindow", "| 我的应用"))
        self.label_7.setText(_translate("MainWindow", "| 移动报表"))
        self.UploadAudioFileButton_2.setText(_translate("MainWindow", "上传音频文件"))
        self.AudioPreprocessingButton_2.setText(_translate("MainWindow", "音频预处理"))
        self.AudioWaveformDisplayButton_2.setText(_translate("MainWindow", "音频波形展示"))
        self.FeatherExtractionButton_2.setText(_translate("MainWindow", "特征提取与结果"))
        self.label_2.setText(_translate("MainWindow", "处理前"))
        self.label_12.setText(_translate("MainWindow", "原始波形图"))
        self.OkButton.setText(_translate("MainWindow", "获取图像"))
        self.label_8.setText(_translate("MainWindow", "降噪："))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = AudioPreprocessing()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
