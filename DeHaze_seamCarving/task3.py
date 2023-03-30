# 暗通道先验图像去雾： J = (I - A) / t + A
# I为有雾图,J为待求解无雾图
# A可取暗通道图中亮度前0.1%的像素点对应原图的位置中亮度最高的像素点（大气光）
# (透射率)t = 1 - w*min( min ( I / A ) )，w通常取0.95
import cv2.cv2 as cv
import numpy as np


def guideFilter(I, p, rds=81, eps=0.001):      # 导向图滤波,I为guidance,p为待处理图
    m_I = cv.boxFilter(I, -1, (rds, rds))
    m_p = cv.boxFilter(p, -1, (rds, rds))
    m_Ip = cv.boxFilter(I*p, -1, (rds, rds))
    m_II = cv.boxFilter(I * I, -1, (rds, rds))
    cov_Ip = m_Ip-m_I*m_p   # 协方差
    var_I = m_II-m_I*m_I    # 方差
    a = cov_Ip/(var_I+eps)
    b = m_p-a*m_I
    m_a = cv.boxFilter(a, -1, (rds, rds))
    m_b = cv.boxFilter(b, -1, (rds, rds))
    return m_a*I+m_b


def getdark(org, rds=7):   # 获取org的暗通道效果,rds为最小值滤波的半径,(15x15效果会不错)
    fir = np.min(org, 2)     # 在RGB通道上取最小值
    # 对fir进行最小值滤波
    sec = cv.erode(fir, np.ones((rds*2+1, rds*2+1)))
    return sec


def AtmLight(org, dark):
    h, w = org.shape[0:2]
    orgsz = h * w
    num = int(0.001 * orgsz)         # 前0.1%
    darktmp = dark.flatten()        # 展平成一维
    orgtmp = org.reshape(orgsz, 3)
    idx = darktmp.argsort()
    A = np.zeros((1, 3))
    maxlht = 0
    for i in range(orgsz-1, orgsz-1-num, -1):
        bgr = orgtmp[idx[i]]
        lht = bgr[0]*0.1 + bgr[1]*0.6 + bgr[2]*0.3
        # lht = np.mean(bgr)
        if lht > maxlht:
            maxlht = lht
            A = orgtmp[idx[i]]
    return A


def gettx(gry, org, A, w=0.95):
    tx = 1 - w * getdark(org / A, 7)
    return guideFilter(gry, tx)


def Dehaze(org):    # 传入的图片为BGR格式,正常处理即可
    gray = cv.cvtColor(org, cv.COLOR_BGR2GRAY)/255
    img = org.copy()/255
    dark = getdark(img)          # 归一化统一操作后再回复uint8
    # 从暗通道中得到A
    A = AtmLight(img, dark)         # 根据效果决定是否对A的上限做出处理
    A = np.clip(A, 0, 0.80)
    tx = gettx(gray, img, A, 0.95)
    tx = np.clip(tx, 0.1, 1)
    rs = np.empty(img.shape, img.dtype)
    for i in range(3):
        rs[:, :, i] = (img[:, :, i]-A[i])/tx + A[i]

    rs += 0.10      # 去雾后通常比较暗,提升亮度让图片更美观
    rs = np.clip(rs, 0, 1)
    rs *= 255
    rs = rs.astype(np.uint8)
    return rs
