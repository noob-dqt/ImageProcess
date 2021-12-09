from PySide2.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import Qt
import cv2.cv2 as cv
# import numpy as np
import sys
import task3
# import task3_seam as cp


class Basic_Edit:
    def __init__(self):
        self.img = None         # opencv的BGR格式
        self.orgimg = None
        self.resimg = None      # 实现
        self.flag = False
        super(Basic_Edit, self).__init__()
        # 设置标题
        self.wd = QUiLoader().load("../UI/task3.ui")
        self.wd.setWindowTitle("图像去雾、裁剪")
        self.wd.openf.triggered.connect(self.openf)
        self.wd.savef.triggered.connect(self.savef)
        self.wd.fogbtn.clicked.connect(self.fogrem)
        # self.wd.cutbtn.clicked.connect(self.cutpic)   # 图形界面会卡，不用了

    def cutpic(self):
        if not self.flag:
            pass
        x = self.wd.wid.text()
        y = self.wd.hei.text()  # 宽度x，高y,注意和图片的shape[0]shape[1]是相反的
        if not x and not y:
            QMessageBox.warning(self.wd, "警告", "请先设置图片宽高")
            return
        x = int(x)
        y = int(y)
        # print(x, type(x))
        # print(y, type(y))
        # 根据设置进行拉伸/裁剪  裁剪50效果比较明显，速度也还算可以！
        self.resimg = cp.maincarving(self.img, x, y)
        self.show_pic()

    def fogrem(self):
        if not self.flag:
            return
        self.resimg = task3.Dehaze(self.img)
        self.show_pic()

    def openf(self):
        filep, _ = QFileDialog.getOpenFileName(self.wd, "选择图片", r"", "*.png *.jpg *.jpeg *.bmp")
        # filep是选取图片的路径
        if not filep:
            QMessageBox.warning(self.wd, "警告", "您未选择图片")
        else:
            self.img = cv.imread(filep)
            self.orgimg = self.img.copy()
            self.resimg = self.img.copy()
            x, y = self.img.shape[:2]
            self.wd.picsize.setText("当前图像像素: "+str(y)+"x"+str(x))
            cv.cvtColor(self.orgimg, cv.COLOR_BGR2RGB, self.orgimg)
            self.flag = True
            self.show_pic()

    def savef(self):
        filep, _ = QFileDialog.getSaveFileName(self.wd, "保存图片", r"", None)
        # filep是图片存储的路径
        # print(filep)
        if not filep:
            QMessageBox.warning(self.wd, "警告", "图片尚未保存")
        else:
            cv.imwrite(filep, self.resimg)

    def show_pic(self):
        if self.flag:
            # 原图保持RGB格式,仅用于展示不做修改
            show_img1 = QImage(self.orgimg, self.orgimg.shape[1], self.orgimg.shape[0], QImage.Format_RGB888)

            cv.cvtColor(self.resimg, cv.COLOR_BGR2RGB, self.resimg)
            show_img2 = QImage(self.resimg, self.resimg.shape[1], self.resimg.shape[0], QImage.Format_RGB888)

            self.wd.orgpic.setPixmap(QPixmap.fromImage(show_img1).scaled(600, 500, aspectMode=Qt.KeepAspectRatio))
            self.wd.orgpic.adjustSize()

            self.wd.respic.setPixmap(QPixmap.fromImage(show_img2).scaled(600, 500, aspectMode=Qt.KeepAspectRatio))
            self.wd.respic.adjustSize()


if __name__ == '__main__':
    app = QApplication([])
    mw = Basic_Edit()
    mw.wd.show()
    sys.exit(app.exec_())
