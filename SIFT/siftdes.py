import cv2.cv2 as cv


def sift_detect(img1, img2):
    sift = cv.xfeatures2d.SIFT_create()
    # 获取关键点
    kp1, feat1 = sift.detectAndCompute(img1, None)
    # print(type(kp1))
    # print(type(des1))
    kp2, feat2 = sift.detectAndCompute(img2, None)   # 获得关键点和特征
    # kps是关键点,所包含的信息有：
    # angle：角度，表示关键点的方向，为了保证方向不变形，SIFT算法通过对关键点周围邻域进行梯度运算，求得该点方向。-1为初值。
    # class_id：当要对图片进行分类时，我们可以用class_id对每个特征点进行区分，未设定时为- 1，需要靠自己设定
    # octave：代表是从金字塔哪一层提取的得到的数据。
    # pt：关键点点的坐标
    # response：响应程度，代表该点强壮大小，更确切的说，是该点角点的程度。
    # size：该点直径的大小
    bf = cv.BFMatcher()
    matches = bf.knnMatch(feat1, feat2, k=2)  # 最近邻匹配
    good = []   # 保存匹配点
    cnt = 0     # 控制匹配点数
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
            cnt += 1
        if cnt >= 50:
            break
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, matchColor=(0, 0, 0), flags=2)     # 不指定颜色
    return img3


if __name__ == "__main__":
    image_a = cv.imread("../images/1.png")
    image_b = cv.imread("../images/2.png")
    img = sift_detect(image_a, image_b)
    cv.imshow("res", img)
    # cv.imwrite("res.png", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
