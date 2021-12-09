# 1.建立高斯金字塔及高斯差分金字塔（模拟不同远近看到的图片）：对一组图片用不同σ做高斯滤波，降采样得到第二组图片（尺寸变小了），
# 再对第二层所有图片用不同σ做卷积，再降采样得到第三层...反复做上述操作得到一个金字塔形状的图片簇（有很多组，每组有多层不同σ卷积得到的结果），
# 再将同一组图片做差分操作（上层减下层）得到高斯差分金字塔
# 2.利用计算得到的高斯差分金字塔计算每组图片的关键点(只有中间层能求，顶层和最下层是没有图片的)；计算关键点第一步需要阈值化去除像素绝对值小于0.5*T/n
# (通常n取0.04，n是从同一组高斯金字塔的n张图片中提取关键点)，第二步，计算极值，在离散化位置中找到极值点，（利用插值法）
# 3.计算关键点方向，获取特征描述符。以特征点为中心找16X16的窗口，计算每个像素的梯度方向，剔除部分较小的梯度值，得到梯度方向的直方图（八个槽），
# 把16X16分成4X4个Block，每个block包含4X4的cell，于是用4X4X8=128维向量来描述图像特征
import numpy as np
import cv2.cv2 as cv


def sift_detect(img1, img2):
    sift = cv.xfeatures2d.SIFT_create()
    # 获取关键点
    kp1, feat1 = sift.detectAndCompute(img1, None)
    kp2, feat2 = sift.detectAndCompute(img2, None)  # 获得关键点和特征(128维向量)
    kps1 = np.float32([kp.pt for kp in kp1])    # 转换成float数组
    kps2 = np.float32([kp.pt for kp in kp2])
    # kp1、2是关键点,所包含的信息有：
    # angle：角度，表示关键点的方向，为了保证方向不变形，SIFT算法通过对关键点周围邻域进行梯度运算，求得该点方向。-1为初值。
    # class_id：当要对图片进行分类时，我们可以用class_id对每个特征点进行区分，未设定时为- 1，需要靠自己设定
    # octave：代表是从金字塔哪一层提取的得到的数据。
    # pt：关键点的坐标
    # response：响应程度，代表该点强壮大小，该点角点的程度。
    # size：该点直径的大小
    bf = cv.BFMatcher()
    rm = bf.knnMatch(feat1, feat2, k=2)  # 最近邻匹配
    # rm的值为Dmatch类型，其中Dmatch类有：
    # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
    # trainIdx：样本图像的特征点描述符下标, 同时也是描述符对应特征点的下标。
    # distance：代表匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
    # print(rm)
    matches = []    # 保存匹配点
    for m in rm:
        if len(m) == 2 and m[0].distance < 0.5 * m[1].distance:     # 0.5是比例，可调节
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) < 4:    # opencv函数至少需要四个匹配点
        print("匹配点太少无法拼接")
        return
    else:
        # 计算匹配的坐标
        pts1 = np.float32([kps1[i] for (_, i) in matches])
        pts2 = np.float32([kps2[i] for (i, _) in matches])
        h, sta = cv.findHomography(pts1, pts2, cv.RANSAC, 4.0)      # 获取视觉变换矩阵
        # 获取完matches、h、sta
        res = cv.warpPerspective(img1, h, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        res[0:img2.shape[0], 0:img2.shape[1]] = img2    # 将图片2传入result图片最左端
        return res
    # return img3


if __name__ == "__main__":
    image_a = cv.imread("../images/pj1.jpg")
    image_b = cv.imread("../images/pj2.jpg")
    # 将sft1拼接在sft2右端
    img = sift_detect(image_a, image_b)
    cv.imshow("res", img)
    # cv.imwrite("res.png", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
