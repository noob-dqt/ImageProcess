import cv2.cv2 as cv
import numpy as np
import math
import copy
import numba as nb


def divid(a):
    if a/2 == 0:
        x1 = x2 = a/2
    else:
        x1 = math.floor(a/2)
        x2 = a - x1
    return -x1, x2


def getval():
    value = np.zeros(256)
    var_temp = 30
    for i in range(0, 255):
        value[i] = math.e ** (-i*i / (2 * var_temp * var_temp))
    return value


@nb.jit()
def gaussian_b0x(a, b):
    judge = 10
    box = []
    x1, x2 = divid(a)
    y1, y2 = divid(b)
    for i in range(x1, x2):
        for j in range(y1, y2):
            t = i*i + j*j
            re = math.e ** (-t/(2*judge*judge))
            box.append(re)
    return box


@nb.jit()
def original(i, j, k, a, b, img):
    x1, x2 = divid(a)
    y1, y2 = divid(b)
    temp = np.zeros(a * b)
    count = 0
    for m in range(x1, x2):
        for n in range(y1, y2):
            if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
                temp[count] = img[i, j, k]
            else:
                temp[count] = img[i + m, j + n, k]
            count += 1
    return temp


@nb.jit()
def bilateral_function(a, b, img, gauss_fun, getval_e):
    x1, x2 = divid(a)
    y1, y2 = divid(b)
    re = np.zeros(a * b)
    timg = copy.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(0, 2):
                temp = original(i, j, k, a, b, timg)
                # print("ave:",ave_temp)
                count = 0
                for m in range(x1, x2):
                    for n in range(y1, y2):
                        if i+m < 0 or i+m > img.shape[0]-1 or j+n < 0 or j+n > img.shape[1]-1:
                            x = img[i, j, k]
                        else:
                            x = img[i+m, j+n, k]
                        t = int(math.fabs(int(x) - int(img[i, j, k])))
                        re[count] = getval_e[t]
                        count += 1
                evalue = np.multiply(re, gauss_fun)
                img[i, j, k] = int(np.average(temp, weights=evalue))
    return img


# if __name__ == "__main__":
#     gauss_new = gaussian_b0x(5, 5)
#     # print(gauss_new)
#     getval_e = getval()
#     timg = cv.imread("E:/Py_work/images/people.jpg")
#     bilateral_img = bilateral_function(5, 5, copy.copy(timg), gauss_new, getval_e)
#     cv.imshow("shuangbian", bilateral_img)
#     cv.imshow("yuantu", timg)
#     cv.imwrite("shuangbian.jpg", bilateral_img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()


