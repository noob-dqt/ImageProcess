import cv2.cv2 as cv
import numpy as np
import numba as nb


def getEnergy(org):
    gra = np.gradient(org)
    tp = np.abs(gra[0]) + np.abs(gra[1])
    tp += 0.5
    tp = tp.astype(np.uint8)
    return tp


@nb.jit()
def getseam(enermap):
    n = enermap.shape[0]
    m = enermap.shape[1]
    path = np.zeros(enermap.shape, np.int32)
    for i in range(1, n):
        for j in range(m):
            mval = 15555
            if j >= 1:
                path[i][j] = -1
                mval = enermap[i][j] + enermap[i-1][j-1]
            if enermap[i][j]+enermap[i-1][j] < mval:
                path[i][j] = 0
                mval = enermap[i][j] + enermap[i-1][j]
            if j < m-1:
                if enermap[i][j] + enermap[i-1][j+1] < mval:
                    path[i][j] = 1
                    mval = enermap[i][j] + enermap[i-1][j+1]
            enermap[i][j] = mval
    # 从enermap[n-1][:]中找到最小的值，并从该往上找到所有删除的点的集合，并返回结果
    minx = np.min(enermap[n-1])
    whe = np.where(minx == enermap[n-1])
    pos = whe[0][0]     # enermap[n-1][pos]为能量最小点
    j = pos
    rs = np.zeros(n, np.int32)
    for i in range(n-1, -1, -1):
        rs[i] = j
        op = path[i][j]
        if op == -1:
            j -= 1
        if op == 1:
            j += 1
    return rs     # 每次返回的结果都应该是长为n的数组，即删除了n行，每行删除一个元素，宽度减小


@nb.jit()
def seam_carving(org, dx):      # org是BGR图
    img = org.copy()
    for i in range(dx):
        n = img.shape[0]
        m = img.shape[1]
        timg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ener_map = getEnergy(timg)
        ener_map = ener_map.astype(np.int32)
        seam = getseam(ener_map)
        # print(seam)
        # 删除路径上的所有像素点
        for j in range(n):
            ty = seam[j]    # 第j行要删除的
            # org[j][ty][:] = [0, 0, 255]
            # 对缝的两端像素取缝上点与各自的平均，让连接更自然(效果并不是很好，放弃)
            # 开始从后往前平移
            for k in range(ty, m-1):
                img[j][k] = img[j][k+1]

        # 平移结束将整个矩阵最后一列删掉，完成一次完整的删除
        img = np.delete(img, m-1, 1)
        if i >= dx-1:
            return img
        # print("return qian:", img.shape)
    return img


@nb.jit()
def getexpseam(enermap, dx):
    n = enermap.shape[0]
    m = enermap.shape[1]
    path = np.zeros(enermap.shape, np.int32)
    for i in range(1, n):
        for j in range(m):
            mval = 15555
            if j >= 1:
                path[i][j] = -1
                mval = enermap[i][j] + enermap[i - 1][j - 1]
            if enermap[i][j] + enermap[i - 1][j] < mval:
                path[i][j] = 0
                mval = enermap[i][j] + enermap[i - 1][j]
            if j < m - 1:
                if enermap[i][j] + enermap[i - 1][j + 1] < mval:
                    path[i][j] = 1
                    mval = enermap[i][j] + enermap[i - 1][j + 1]
            enermap[i][j] = mval
    # 最后的处理有所不同，同时选出dx条缝seam=>[dx][n]
    ascidx = enermap[n-1].argsort()
    rs = np.zeros((dx, n), np.int32)
    for k in range(dx-1, -1, -1):
        j = ascidx[k]
        # j = ascidx[m-1-k]
        for i in range(n - 1, -1, -1):
            rs[k][i] = j
            op = path[i][j]
            if op == -1:
                j -= 1
            if op == 1:
                j += 1
    return rs


@nb.jit()
def seamexpand(org, dx):
    img = org.copy()
    timg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ener_map = getEnergy(timg)
    ener_map = ener_map.astype(np.int32)
    seam = getexpseam(ener_map, dx)
    # seam==>dx * n 在每条缝旁边复制一份
    for i in range(dx):
        n = img.shape[0]
        m = img.shape[1]
        ist = np.zeros((n, 3), np.int32)  # 插入的缝
        img = np.insert(img, m, ist, 1)     # 尾部插入一条缝
        for j in range(n):
            pos = seam[i][j]
            for k in range(img.shape[1]-1, pos-1, -1):
                img[j][k] = img[j][k-1]
            img[j][pos][0] = int((int(img[j][pos][0]) + int(img[j][pos + 1][0])) / 2)
            img[j][pos][1] = int((int(img[j][pos][1]) + int(img[j][pos + 1][1])) / 2)
            img[j][pos][2] = int((int(img[j][pos][2]) + int(img[j][pos + 1][2])) / 2)
        if i >= dx-1:
            return img
    return img

# ****方案改为每次都选取最小线得到得seam会重复****
# @nb.jit()
# def seamexpand(org, dx):
#     img = org.copy()
#     # seam==>dx * n 在每条缝旁边复制一份
#     for i in range(dx):
#         n = img.shape[0]
#         m = img.shape[1]
#         timg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         ener_map = getEnergy(timg)
#         ener_map = ener_map.astype(np.int32)
#         seam = getseam(ener_map)
#         ist = np.zeros((n, 3), np.int32)  # 插入的缝
#         img = np.insert(img, m, ist, 1)     # 尾部插入一条缝
#         for j in range(n):
#             pos = seam[j]
#             for k in range(img.shape[1]-1, pos-1, -1):
#                 img[j][k] = img[j][k-1]
#         if i >= dx-1:
#             return img
#     return img


def maincarving(org, w, h):
    img = org.copy()
    n = img.shape[0]
    m = img.shape[1]
    if w <= m:
        img = seam_carving(img, m - w)
    else:       # 加宽图像
        img = seamexpand(img, w - m)
    if h <= n:
        timg = np.transpose(img, (1, 0, 2))     # 转置
        timg = seam_carving(timg, n - h)
        img = np.transpose(timg, (1, 0, 2))
    else:
        timg = np.transpose(img, (1, 0, 2))  # 转置
        timg = seamexpand(timg, h - n)
        img = np.transpose(timg, (1, 0, 2))
    return img


x = cv.imread("E:/Py_work/images/cut.jpg")
# ans = maincarving(x, x.shape[1], x.shape[0] - 100)       # 高度方向
ans = maincarving(x, x.shape[1] - 100, x.shape[0])       # 宽度方向
# ******加宽********
# ans = maincarving(x, x.shape[1], x.shape[0] + 100)       # 高度方向
# ans = maincarving(x, x.shape[1] + 100, x.shape[0])       # 宽度方向
cv.imshow("org", x)
cv.imshow("res", ans)
cv.waitKey(0)
cv.destroyAllWindows()
