from PySide2.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QImage, QPixmap
import cv2.cv2 as cv
# import numpy as np
import sys
import task1
from PySide2.QtCore import Qt


class Basic_Edit:
    def __init__(self):
        self.img = None         # opencv的BGR格式
        self.tmpimg = None      # 用于显示
        self.flag = False
        super(Basic_Edit, self).__init__()
        # 设置标题
        self.wd = QUiLoader().load("../UI/task1.ui")
        self.wd.setWindowTitle("DQTの丑图秀秀")
        # self.wd.resize(800, 600)
        self.wd.bar1.valueChanged.connect(self.cal)
        self.wd.bar2.valueChanged.connect(self.cal)
        self.wd.bar3.valueChanged.connect(self.cal)     # 三个bar触发同个信号槽
        self.wd.openf.triggered.connect(self.openf)
        self.wd.savef.triggered.connect(self.savef)

    def openf(self):
        filep, _ = QFileDialog.getOpenFileName(self.wd, "选择图片", r"", "*.png *.jpg *.jpeg *.bmp")
        # filep是选取图片的路径
        # print(filep)
        if not filep:
            QMessageBox.warning(self.wd, "警告", "您未选择图片")
        else:
            self.img = cv.imread(filep)
            self.tmpimg = self.img.copy()
            self.flag = True
            self.show_pic()

    def savef(self):
        filep, _ = QFileDialog.getSaveFileName(self.wd, "保存图片", r"", None)
        # filep是图片存储的路径
        # print(filep)
        if not filep:
            QMessageBox.warning(self.wd, "警告", "图片尚未保存")
        else:
            cv.imwrite(filep, self.tmpimg)

    def show_pic(self):
        if self.flag:
            # 第三个参数放self.tmpimg就可以解决加载不了大像素图片的问题，但是有些图像颜色加载进来后颜色发生变化,因此需要绘制图形后再进行一次空间转换
            # cv.imshow("cmp", self.tmpimg)
            cv.cvtColor(self.tmpimg, cv.COLOR_BGR2RGB, self.tmpimg)
            show_img = QImage(self.tmpimg, self.tmpimg.shape[1], self.tmpimg.shape[0], QImage.Format_RGB888)

            # show_img = cv.cvtColor(self.tmpimg, cv.COLOR_BGR2RGB)
            # show_img = QImage(show_img, show_img.shape[1], show_img.shape[0], QImage.Format_RGB888)
            # 很奇怪，只要打开像素较大的图片程序就会崩溃
            # self.wd.pic.setScaledContents(True)     # 图片自适应Label

            self.wd.pic.setPixmap(QPixmap.fromImage(show_img).scaled(800, 600, aspectMode=Qt.KeepAspectRatio))
            self.wd.pic.adjustSize()
            cv.cvtColor(self.tmpimg, cv.COLOR_RGB2BGR, self.tmpimg)     # 必须在此绘制结束后把tmpimg转回BGR否则红蓝通道会翻转

    def cal(self):
        s1 = self.wd.bar1.value()
        s2 = self.wd.bar2.value()
        s3 = self.wd.bar3.value()
        # print(s1, s2, s3)
        # print(s1/100.0, s2/100.0, s3/100.0)
        ads1 = str(s1) + "%"
        ads2 = str(s2) + "%"
        ads3 = str(s3) + "%"
        self.wd.pro_l.setText(ads1)
        self.wd.pro_c.setText(ads2)
        self.wd.pro_s.setText(ads3)
        if self.flag:       # 无图不操作
            self.tmpimg = task1.calcu.lightness_(self.img, s1 / 100.0)
            self.tmpimg = task1.calcu.contrast_rt(self.tmpimg, s2 / 100.0)
            self.tmpimg = task1.calcu.saturation(self.tmpimg, s3 / 100.0)
            self.show_pic()


if __name__ == '__main__':
    app = QApplication([])
    mw = Basic_Edit()
    mw.wd.show()
    # app.exec_()
    sys.exit(app.exec_())
