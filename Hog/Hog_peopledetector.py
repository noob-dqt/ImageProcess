import cv2.cv2 as cv


# 图像 resize=>128X64,8x8作为一个cell(共8x16个cell)，计算图像梯度，根据梯度方向将0~180分为9份，
# 将梯度对应角度的赋值加入相应的直方图里，将4个cell作为一个block，合并直方图，得到一个7x15x9x4=3780维的向量——即Hog特征描述的方法
if __name__ == '__main__':
    img = cv.imread("../images/xr.jpg")
    hog = cv.HOGDescriptor()
    # winSize, blockSize, blockStride, cellSize, nbins
    # 分别是窗口大小、block大小、block步长、cell大小、bin的取值
    org = img.copy()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    rects, val = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.10, useMeanshiftGrouping=False)  # 在多尺度上寻找
    # print(rects)
    # winStride:Hog检测移动步长
    for (x, y, w, h) in rects:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.imshow("org", org)
    cv.imshow("res", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
