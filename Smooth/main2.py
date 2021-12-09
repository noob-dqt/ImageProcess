from PySide2.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QImage, QPixmap
import cv2.cv2 as cv
# import numpy as np
import task2 as t2
import sys
from PySide2.QtCore import Qt


class Basic_Edit:
    def __init__(self):
        self.img = None         # opencv的BGR格式
        self.orgimg = None
        self.tmpimg = None      # 自己实现
        self.stdimg = None      # opencv库处理
        self.flag = False
        super(Basic_Edit, self).__init__()
        # 设置标题
        self.wd = QUiLoader().load("../UI/task2.ui")
        self.wd.setWindowTitle("滤波操作")
        self.wd.openf.triggered.connect(self.openf)
        self.wd.savef.triggered.connect(self.savef)
        self.wd.boxf.clicked.connect(self.boxcal)
        self.wd.medf.clicked.connect(self.medcal)
        self.wd.maxf.clicked.connect(self.maxcal)
        self.wd.minf.clicked.connect(self.mincal)
        self.wd.guasf.clicked.connect(self.guascal)
        self.wd.bilf.clicked.connect(self.bilcal)

    def bilcal(self):
        if not self.flag:
            return
        self.tmpimg = t2.self_filt().bilf(self.img, 5, 75, 75)
        self.stdimg = t2.Cv_Lib().Bil(self.img, 5, 75, 75)
        self.show_pic()

    def guascal(self):
        if not self.flag:
            return
        self.tmpimg = t2.self_filt().guasf(self.img, 5, 1)
        self.stdimg = t2.Cv_Lib().Gaus(self.img, 5, 1)
        self.show_pic()

    def maxcal(self):
        if not self.flag:
            return
        self.tmpimg = t2.self_filt().maxf(self.img, 5)  # 指定滤波核大小
        self.stdimg = t2.Cv_Lib().Max(self.img, 5)
        self.show_pic()

    def mincal(self):
        if not self.flag:
            return
        self.tmpimg = t2.self_filt().minf(self.img, 5)  # 指定滤波核大小
        self.stdimg = t2.Cv_Lib().Min(self.img, 5)
        self.show_pic()

    def boxcal(self):
        if not self.flag:
            return
        selft = t2.self_filt()
        selfs = t2.Cv_Lib()
        self.tmpimg = selft.boxf(self.img, 5)    # 指定滤波核大小
        self.stdimg = selfs.Box(self.img, 5)
        self.show_pic()

    def medcal(self):
        if not self.flag:
            return
        self.tmpimg = t2.self_filt().medf(self.img, 5)    # 指定滤波核大小
        self.stdimg = t2.Cv_Lib().Med(self.img, 5)
        self.show_pic()

    def openf(self):
        filep, _ = QFileDialog.getOpenFileName(self.wd, "选择图片", r"", "*.png *.jpg *.jpeg *.bmp")
        # filep是选取图片的路径
        if not filep:
            QMessageBox.warning(self.wd, "警告", "您未选择图片")
        else:
            self.img = cv.imread(filep)
            cv.cvtColor(self.img, cv.COLOR_BGR2RGB, self.img)
            self.orgimg = self.img.copy()
            self.tmpimg = self.img.copy()
            self.stdimg = self.img.copy()
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
            # cv.cvtColor(self.tmpimg, cv.COLOR_BGR2RGB, self.tmpimg)
            show_img2 = QImage(self.tmpimg, self.tmpimg.shape[1], self.tmpimg.shape[0], QImage.Format_RGB888)

            # cv.cvtColor(self.stdimg, cv.COLOR_BGR2RGB, self.stdimg)
            show_img1 = QImage(self.stdimg, self.stdimg.shape[1], self.stdimg.shape[0], QImage.Format_RGB888)

            # cv.cvtColor(self.orgimg, cv.COLOR_BGR2RGB, self.orgimg)
            show_img3 = QImage(self.orgimg, self.orgimg.shape[1], self.orgimg.shape[0], QImage.Format_RGB888)

            self.wd.stdpic.setPixmap(QPixmap.fromImage(show_img1).scaled(600, 400, aspectMode=Qt.KeepAspectRatio))
            self.wd.stdpic.adjustSize()

            self.wd.slfpic.setPixmap(QPixmap.fromImage(show_img2).scaled(600, 400, aspectMode=Qt.KeepAspectRatio))
            self.wd.slfpic.adjustSize()

            self.wd.orgpic.setPixmap(QPixmap.fromImage(show_img3).scaled(600, 400, aspectMode=Qt.KeepAspectRatio))
            self.wd.orgpic.adjustSize()

            # cv.cvtColor(self.tmpimg, cv.COLOR_RGB2BGR, self.tmpimg)     # 必须在此绘制结束后把tmpimg转回BGR否则红蓝通道会翻转
            # cv.cvtColor(self.stdimg, cv.COLOR_RGB2BGR, self.stdimg)
            # cv.cvtColor(self.orgimg, cv.COLOR_RGB2BGR, self.orgimg)


if __name__ == '__main__':
    app = QApplication([])
    mw = Basic_Edit()
    mw.wd.show()
    sys.exit(app.exec_())
