# 均值(Box) 高斯(Gaussian) 中值(Median) 双边(Bilateral) 最大值&最小值滤波
import numpy as np
import cv2.cv2 as cv
import numba as nb
import math
from Smooth import test as bils
import copy as cp

class Cv_Lib:
    def Box(self, img, rds):
        return cv.blur(img, (rds, rds))

    def Gaus(self, img, rds, sgm):
        return cv.GaussianBlur(img, (rds, rds), sgm)

    def Med(self, img, rds):
        return cv.medianBlur(img, rds)

    def Bil(self, img, rds, sgms, sgmc):
        return cv.bilateralFilter(img, rds, sgms, sgmc)

    def Max(self, img, rds):
        return img

    def Min(self, img, rds):
        return img


class self_filt:
    @nb.jit()
    def calkernel(self, timg, rds, ker):
        img = timg.copy()
        # img = img.astype(np.float32)
        nums = rds * rds
        rds = int(rds / 2)
        n = img.shape[0]
        m = img.shape[1]
        for i in range(rds, n - rds):
            for j in range(rds, m - rds):
                for k in range(3):
                    st = 0
                    for x in range(i-rds, i+rds+1):
                        for y in range(j-rds, j+rds+1):
                            st += ker[x-i+rds][y-j+rds] * timg[x][y][k]
                    if st > 255:
                        st = 255
                    img[i][j][k] = st
        return img

    @nb.jit()
    def boxf(self, timg, rds):
        img = timg.copy()
        nums = rds * rds
        rds = int(rds / 2)
        # print(rds)
        n = img.shape[0]
        m = img.shape[1]
        # if i < rds or i >= n-rds or j < rds or j > m-rds:
        #     img[i][j][:] -= 25
        #     img[i][j][:][img[i][j][:] < 0] = 0
        #     continue
        for i in range(rds, n-rds):
            for j in range(rds, m-rds):
                for k in range(3):
                    img[i][j][k] = int(np.mean(timg[i - rds:i + rds + 1, j - rds:j + rds + 1, k]))
                    # s = 0
                    # for x in range(i-rds, i+rds+1):
                    #     for y in range(j-rds, j+rds+1):
                    #         s += timg[x][y][k]
                    # img[i][j][k] = s / nums
        # img = img.astype(np.uint8)
        return img

    @nb.jit()
    def maxf(self, timg, rds):
        img = timg.copy()
        rds = int(rds / 2)
        n = img.shape[0]
        m = img.shape[1]
        for i in range(rds, n - rds):
            for j in range(rds, m - rds):
                for k in range(3):
                    img[i][j][k] = np.max(timg[i - rds:i + rds + 1, j - rds:j + rds + 1, k])
        return img

    @nb.jit()
    def minf(self, timg, rds):
        img = timg.copy()
        rds = int(rds / 2)
        n = img.shape[0]
        m = img.shape[1]
        for i in range(rds, n - rds):
            for j in range(rds, m - rds):
                for k in range(3):
                    img[i][j][k] = np.min(timg[i - rds:i + rds + 1, j - rds:j + rds + 1, k])
        return img

    @nb.jit()
    def guasf(self, timg, rds, sgm):    # 先生成rds*rds的权值矩阵，再对所有所有像素进行加权操作
        # 计算算子部分
        gusMat = np.zeros([rds, rds], np.float32)
        th = (rds - 1)/2
        for i in range(rds):
            for j in range(rds):
                tp = math.pow(i-th, 2) + math.pow(j - th, 2)
                gusMat[i][j] = math.exp(-tp/(2*math.pow(sgm, 2)))
        sm = np.sum(gusMat)
        gusMat /= sm       # 归一化

        # 图像计算,调用calkernel
        return self.calkernel(timg, rds, gusMat)

    @nb.jit()
    def medf(self, timg, rds):
        img = timg.copy()
        rds = int(rds / 2)
        n = img.shape[0]
        m = img.shape[1]
        for i in range(rds, n - rds):
            for j in range(rds, m - rds):
                for k in range(3):
                    img[i][j][k] = np.median(timg[i-rds:i+rds+1, j-rds:j+rds+1, k])
        return img

# bil************************
#     def divd(self, a):
#         if a / 2 == 0:
#             x1 = x2 = a / 2
#         else:
#             x1 = math.floor(a / 2)
#             x2 = a - x1
#         return -x1, x2
#
#     def getval(self):
#         value = np.zeros(256)
#         var_temp = 30
#         for i in range(0, 255):
#             value[i] = math.exp((-i * i / (2 * var_temp * var_temp)))
#         return value
#
#     @nb.jit()
#     def original(self, i, j, k, a, b, img):
#         x1, x2 = self.divd(a)
#         y1, y2 = self.divd(b)
#         temp = np.zeros(a * b)
#         count = 0
#         for m in range(x1, x2):
#             for n in range(y1, y2):
#                 if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
#                     temp[count] = img[i, j, k]
#                 else:
#                     temp[count] = img[i + m, j + n, k]
#                 count += 1
#         return temp
#
#     @nb.jit()
#     def bilateral_function(self, a, b, img, gauss_fun, getval_e):
#         x1, x2 = self.divd(a)
#         y1, y2 = self.divd(b)
#         re = np.zeros(a*b)
#         tmpimg = cp.copy(img)
#         # tmpimg = img.copy()
#         for i in range(img.shape[0]):
#             for j in range(img.shape[1]):
#                 for k in range(0, 2):
#                     temp = self.original(i, j, k, a, b, tmpimg)
#                     count = 0
#                     for m in range(x1, x2):
#                         for n in range(y1, y2):
#                             if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
#                                 x = img[i, j, k]
#                             else:
#                                 x = img[i + m, j + n, k]
#                             t = int(math.fabs(int(x) - int(img[i, j, k])))
#                             re[count] = getval_e[t]
#                             count += 1
#                     evalue = np.multiply(re, gauss_fun)
#                     img[i, j, k] = int(np.average(temp, weights=evalue))
#         return img

# *************************
    @nb.jit()
    def bilf(self, timg, rds, sgms, sgmc):
        gusmat = bils.gaussian_b0x(rds, rds)
        colors = bils.getval()
        return bils.bilateral_function(rds, rds, cp.copy(timg), gusmat, colors)


