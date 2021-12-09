import cv2.cv2 as cv
import numpy as np
# 实现三个功能：对比度、亮度、色彩饱和度
# HLS 色相、亮度、饱和

class calcu:
    # def __init__(self, pth):
        # pass
    # 亮度
    def lightness_(t_img, factor):  # factor值于-1~40之间(为1最高亮度提高到2倍,factor>=40的时候全白)
        cg = int(factor * 255)
        # hls_img = t_img.astype(np.float32)
        # hls_img /= 255.0  # 用int16类型会出现莫名奇妙的白色溢出...还是转换一下比较稳妥~
        # hls_img = cv.cvtColor(hls_img, cv.COLOR_BGR2HLS)  # 色彩空间转换不支持float16类型
        # # print(hls_img)
        # hls_img[:, :, 1] *= (factor + 1.0)
        # hls_img[:, :, 1][hls_img[:, :, 1] > 1] = 1
        # # hls_img[:, :, 1][hls_img[:, :, 1] < 0] = 0
        # t_img = cv.cvtColor(hls_img, cv.COLOR_HLS2BGR)
        # # cv.imshow("", t_img)
        # t_img *= 255
        # t_img = t_img.astype(np.uint8)  # 转回原格式,避免对其余操作造成影响
        # return t_img
        hls_img = t_img.astype(np.int16)
        for i in range(3):
            hls_img[:, :, i] += cg
            hls_img[:, :, i][hls_img[:, :, i] > 255] = 255
            hls_img[:, :, i][hls_img[:, :, i] < 0] = 0
        t_img = hls_img
        # t_img = cv.cvtColor(hls_img, cv.COLOR_HLS2BGR)
        # cv.imshow("", t_img)
        # t_img *= 255
        t_img = t_img.astype(np.uint8)  # 转回原格式,避免对其余操作造成影响
        return t_img


    # 色彩饱和度
    def saturation(t_img, factor):  # factor值于-1~1之间(factor=-1时失去所有色彩,factor过大会导致图片颜色变多)
        hls_img = t_img.astype(np.float32)
        hls_img /= 255.0  # 用int16类型会出现莫名奇妙的白色溢出...还是转换一下比较稳妥~
        hls_img = cv.cvtColor(hls_img, cv.COLOR_BGR2HLS)  # 色彩空间转换不支持float16类型
        # print(hls_img)
        hls_img[:, :, 2] *= (factor + 1.0)
        hls_img[:, :, 2][hls_img[:, :, 2] > 1] = 1
        # hls_img[:, :, 1][hls_img[:, :, 1] < 0] = 0
        t_img = cv.cvtColor(hls_img, cv.COLOR_HLS2BGR)
        # cv.imshow("", t_img)
        t_img *= 255
        t_img = t_img.astype(np.uint8)  # 转回原格式,避免对其余操作造成影响
        return t_img

    # 对比度
    def contrast_rt(t_img, factor):  # 1>=factor>=0,factor=1,小于平均值全变黑
        t_img = t_img.astype(np.float32)
        # print(t_img)
        # shd = np.median(t_img[:, :, i])  # 第一种方案：取中位数
        # shd = np.mean(t_img[:, :, i])         # 第二种方案：取平均数
        # print(shd)
        if factor >= 0:
            for i in range(3):
                shd = np.median(t_img[:, :, i])  # 第一种方案：取中位数
                t_img[:, :, i][t_img[:, :, i] >= shd] *= (1.0 + factor)
                t_img[:, :, i][t_img[:, :, i] < shd] *= (1.0 - factor)
                # 对于正好等于均值的像素点不做操作
                t_img[:, :, i][t_img[:, :, i] > 255] = 255
                t_img[:, :, i][t_img[:, :, i] < 0] = 0
                # print(t_img[:, :, i])
        else:   # 减小对比度
            for i in range(3):
                shd = np.median(t_img[:, :, i])  # 第一种方案：取中位数
                cg = factor * shd * 0.35
                # t_img[:, :, i][t_img[:, :, i] < 1.0] += 25      # 先让黑色大于1
                t_img[:, :, i][t_img[:, :, i] < shd] -= cg
                t_img[:, :, i][t_img[:, :, i] > shd] += cg

                # 对于正好等于均值的像素点不做操作
                # t_img[:, :, i][t_img[:, :, i] < shd] = shd
                # t_img[:, :, i][t_img[:, :, i] > shd] = shd
                # t_img[:, :, i][t_img[:, :, i] > 255] = shd
                # t_img[:, :, i][t_img[:, :, i] < 0] = shd
                # print(t_img[:, :, i])

        t_img = t_img.astype(np.uint8)
        # cv.imshow("dui", t_img)
        return t_img




# org = cv.imread("E:\\Py_work\\images\\p2.jpg")
# tmp = lightness_(tmp, 1)
# tmp = saturation(tmp, 1)
# tmp = contrast_rt(org, 0.18)
# print(tmp)
# cv.imshow("dui", tmp)
# cv.imshow("org", org)

# cv.waitKey(0)
# cv.destroyAllWindows()
