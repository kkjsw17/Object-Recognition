import numpy as np
from math import pi, sqrt
import cv2
import collections


class Processor:
    def __init__(self, src):
        self.src = src
        self.Height = src.shape[0]
        self.Width = src.shape[1]
        self.Channel = src.shape[2]
        self.colorList = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                          (255, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
                          (128, 0, 128), (0, 128, 128), (75, 255, 32), (200, 100, 50), (80, 40, 222),
                          (208, 94, 32), (102, 50, 255), (80, 50, 150), (10, 90, 222), (160, 30, 70),
                          (48, 50, 99), (80, 110, 222), (111, 111, 222), (222, 30, 55), (70, 150, 160),
                          (210, 50, 88), (99, 179, 99), (128, 60, 30), (30, 60, 128), (30, 128, 60),
                          (128, 30, 60), (60, 128, 30), (60, 30, 128), (210, 88, 50), (88, 210, 50),
                          (88, 50, 210), (50, 88, 210), (50, 210, 88), (222, 110, 80), (222, 80, 110),
                          (110, 80, 222), (110, 222, 80), (150, 50, 80), (150, 80, 50), (50, 80, 150)]

    def gaussianFiltering(self, shape, sigma):
        dst = np.zeros((self.Height, self.Width, self.Channel), dtype='uint8')
        s = int((shape[0] - 1) / 2)
        t = int((shape[1] - 1) / 2)
        x, y = np.ogrid[-s:s + 1, -t:t + 1]
        gaussian_kernel = np.exp(-(x * x + y * y) / sigma ** 2) / (2. * pi * sigma ** 2)
        gaussian_kernel /= np.sum(gaussian_kernel)

        pad_src = np.zeros((self.Height + s * 2, self.Width + t * 2, self.Channel), dtype='uint8')
        pad_src[s:s + self.Height, t:t + self.Width] = self.src

        for i in range(self.Height):
            for j in range(self.Width):
                for k in range(self.Channel):
                    dst[i][j][k] = np.sum(pad_src[i:i + shape[0], j:j + shape[1], k] * gaussian_kernel)

        self.src = dst

    def histEqulization(self):
        gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        histogram = np.zeros(256, dtype='int64')
        after_histogram = np.zeros(256, dtype='int64')
        sum_hist = np.zeros(256, dtype='int64')

        for row in gray:
            for pixel in row:
                histogram[pixel] += 1

        summ = 0.0
        scale_factor = 255.0 / (width * height)
        for i, h in enumerate(histogram):
            summ += h
            sum_hist[i] = (summ * scale_factor) + 0.5

        for row in gray:
            for pixel in row:
                pixel = sum_hist[pixel]
                after_histogram[pixel] += 1

        hsv = cv2.cvtColor(self.src, cv2.COLOR_BGR2HSV)
        h, w, c = hsv.shape
        for i in range(0, h):
            for j in range(0, w):
                hsv[i][j][2] = gray[i][j]

        self.src = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def motionEstimation(self, prev, step=16):
        now_frame = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(now_frame, prev_frame, None, 0.5, 3, 13, 3, 5, 1.1, 0)
        gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        return fx, fy

    def detectSkin(self, minh, maxh, mins, maxs):
        # yCrCb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        hsv = cv2.cvtColor(self.src, cv2.COLOR_BGR2HSV)

        for i in range(self.Height):
            for j in range(self.Width):
                # CrCb = np.array(yCrCb[i, j, 1], yCrCb[i, j, 2])
                # D = (CrCb - u).T * np.cov(CrCb - u)
                # print(D)
                # if D[0] + D[1] > B:
                #     frame[i, j] = (0, 0, 0)
                h = hsv[i, j, 0]
                s = hsv[i, j, 1]
                if not ((minh <= h <= maxh) and (mins <= s <= maxs)):
                    self.src[i, j] = (0, 0, 0)

    def connectedComponentLabeling(self):
        gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), dtype='uint8')
        morp = cv2.dilate(cv2.erode(otsu, kernel, iterations=1), kernel, iterations=1)

        pad_img = np.pad(morp, ((1, 1), (1, 1)), 'constant', constant_values=0)
        comp = np.zeros((self.Height + 2, self.Width + 2))
        cmpcnt = 0
        eqvTable = {}
        cmp_tmp_list = []

        for i in range(self.Height + 2):
            for j in range(self.Width + 2):
                if pad_img[i, j] != 0:
                    flag = False
                    if pad_img[i - 1, j - 1] != 0 and not flag:
                        comp[i, j] = comp[i - 1, j - 1]
                        flag = True
                    if pad_img[i - 1, j] != 0 and not flag:
                        comp[i, j] = comp[i - 1, j]
                        flag = True
                    if pad_img[i - 1, j + 1] != 0 and not flag:
                        comp[i, j] = comp[i - 1, j + 1]
                        flag = True
                    if pad_img[i, j - 1] != 0 and not flag:
                        comp[i, j] = comp[i, j - 1]
                        flag = True

                    if comp[i - 1, j - 1] > comp[i][j]:
                        eqvTable[comp[i - 1, j - 1]] = comp[i][j]
                    if comp[i - 1, j] > comp[i][j]:
                        eqvTable[comp[i - 1, j]] = comp[i][j]
                    if comp[i - 1, j + 1] > comp[i][j]:
                        eqvTable[comp[i - 1, j + 1]] = comp[i][j]
                    if comp[i, j - 1] > comp[i][j]:
                        eqvTable[comp[i, j - 1]] = comp[i][j]

                    if not flag:
                        cmpcnt += 1
                        comp[i][j] = cmpcnt
                        cmp_tmp_list.append(HumanComponent(cmpcnt))

        orderedEqvTable = collections.OrderedDict(sorted(eqvTable.items(), reverse=True))

        for key in orderedEqvTable.keys():
            for i in range(self.Height + 2):
                for j in range(self.Width + 2):
                    if comp[i, j] == key:
                        comp[i, j] = eqvTable[key]
                        cmp_tmp_list = [cmp for cmp in cmp_tmp_list if cmp.num != key]

        base_size = (self.Width / 20) * (self.Height / 20)
        base_size2 = (self.Width / 2) * (self.Height / 2)
        cmp_list = []
        for cmp in cmp_tmp_list:
            loc = np.argwhere(comp == cmp.num)
            cmp.setLocation(loc)

            if base_size <= cmp.actual_area <= base_size2:
                if 0.45 <= cmp.circularity <= 0.7:
                    print(cmp.circularity)
                    cmp_list.append(cmp)

        ret_img = np.zeros((self.Height, self.Width, 3), dtype='uint8')

        for i in range(self.Height):
            for j in range(self.Width):
                cnt = 0
                for cmp in cmp_list:
                    if comp[i + 1, j + 1] == cmp.num:
                        ret_img[i, j] = self.colorList[cnt]
                    cnt += 1

        for cmp in cmp_list:
            for i in range(cmp.xstart, cmp.xstop + 1):
                ret_img[cmp.ystart, i] = (0, 0, 255)
                ret_img[cmp.ystop, i] = (0, 0, 255)
            for i in range(cmp.ystart + 1, cmp.ystop):
                ret_img[i, cmp.xstart] = (0, 0, 255)
                ret_img[i, cmp.xstop] = (0, 0, 255)

        self.src = ret_img

    def connectedComponentLabeling_vehicle(self):
        gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((7, 7), dtype='uint8')
        morp = cv2.dilate(gray, kernel, iterations=1)

        pad_img = np.pad(morp, ((1, 1), (1, 1)), 'constant', constant_values=0)
        comp = np.zeros((self.Height + 2, self.Width + 2))
        cmpcnt = 0
        eqvTable = {}
        cmp_tmp_list = []

        for i in range(self.Height + 2):
            for j in range(self.Width + 2):
                if pad_img[i, j] != 0:
                    flag = False
                    if pad_img[i - 1, j - 1] != 0 and not flag:
                        comp[i, j] = comp[i - 1, j - 1]
                        flag = True
                    if pad_img[i - 1, j] != 0 and not flag:
                        comp[i, j] = comp[i - 1, j]
                        flag = True
                    if pad_img[i - 1, j + 1] != 0 and not flag:
                        comp[i, j] = comp[i - 1, j + 1]
                        flag = True
                    if pad_img[i, j - 1] != 0 and not flag:
                        comp[i, j] = comp[i, j - 1]
                        flag = True

                    if comp[i - 1, j - 1] > comp[i][j]:
                        eqvTable[comp[i - 1, j - 1]] = comp[i][j]
                    if comp[i - 1, j] > comp[i][j]:
                        eqvTable[comp[i - 1, j]] = comp[i][j]
                    if comp[i - 1, j + 1] > comp[i][j]:
                        eqvTable[comp[i - 1, j + 1]] = comp[i][j]
                    if comp[i, j - 1] > comp[i][j]:
                        eqvTable[comp[i, j - 1]] = comp[i][j]

                    if not flag:
                        cmpcnt += 1
                        comp[i][j] = cmpcnt
                        cmp_tmp_list.append(VehicleComponent(cmpcnt))

        orderedEqvTable = collections.OrderedDict(sorted(eqvTable.items(), reverse=True))

        for key in orderedEqvTable.keys():
            for i in range(self.Height + 2):
                for j in range(self.Width + 2):
                    if comp[i, j] == key:
                        comp[i, j] = eqvTable[key]
                        cmp_tmp_list = [cmp for cmp in cmp_tmp_list if cmp.num != key]

        cmp_list = []
        for cmp in cmp_tmp_list:
            loc = np.argwhere(comp == cmp.num)
            cmp.setLocation(loc)
            if cmp.ratio >= 0.5 and cmp.convexity >= 0.45:
                cmp_list.append(cmp)

        ret_img = np.zeros((self.Height, self.Width, 3), dtype='uint8')

        for i in range(self.Height):
            for j in range(self.Width):
                cnt = 0
                for cmp in cmp_list:
                    if cnt >= len(self.colorList):
                        break
                    if comp[i + 1, j + 1] == cmp.num:
                        ret_img[i, j] = self.colorList[cnt]
                    cnt += 1

        for cmp in cmp_list:
            for i in range(cmp.xstart, cmp.xstop + 1):
                ret_img[cmp.ystart, i] = (0, 0, 255)
                ret_img[cmp.ystop, i] = (0, 0, 255)
            for i in range(cmp.ystart + 1, cmp.ystop):
                ret_img[i, cmp.xstart] = (0, 0, 255)
                ret_img[i, cmp.xstop] = (0, 0, 255)

        self.src = ret_img


class HumanComponent:
    def __init__(self, num):
        self.num = num
        self.ystart = 0
        self.xstart = 0
        self.ystop = 0
        self.xstop = 0
        self.area = 0
        self.actual_area = 0
        self.circularity = 0

    def setLocation(self, location):
        (self.ystart, self.xstart), (self.ystop, self.xstop) = location.min(0) - 1, location.max(0) - 1

        self.area = (self.ystop - self.ystart + 1) * (self.xstop - self.xstart + 1)

        border = []
        border_check_list = []
        prev_loc = None

        for i in range(self.xstart, self.xstop + 1):
            border.append((self.ystart, i))
        border_check_list.append(self.ystart)

        for loc in location:
            if loc[0] == self.ystart:
                break
            if border_check_list.count(loc[0]) == 0:
                if prev_loc is not None:
                    border.append(prev_loc)
                border.append((loc[0], loc[1]))
                border_check_list.append(loc[0])
            prev_loc = (loc[0], loc[1])

        for i in range(self.xstart, self.xstop + 1):
            border.append((self.ystop, i))

        max_diam = 0
        for i in range(0, len(border)):
            for j in range(i, len(border)):
                diam = sqrt((border[i][0] - border[j][0]) ** 2 + (border[i][1] - border[j][1]) ** 2)
                if diam > max_diam:
                    max_diam = diam

        self.actual_area = location.size
        self.circularity = (4 * self.area) / (pi * max_diam ** 2)


class VehicleComponent:
    def __init__(self, num):
        self.num = num
        self.ystart = 0
        self.xstart = 0
        self.ystop = 0
        self.xstop = 0
        self.convexity = 0
        self.ratio = 0

    def setLocation(self, location):
        (self.ystart, self.xstart), (self.ystop, self.xstop) = location.min(0) - 1, location.max(0) - 1

        border = []
        border_check_list = []
        prev_loc = None

        for i in range(self.xstart, self.xstop + 1):
            border.append((self.ystart, i))
        border_check_list.append(self.ystart)

        for loc in location:
            if loc[0] == self.ystart:
                break
            if border_check_list.count(loc[0]) == 0:
                if prev_loc is not None:
                    border.append(prev_loc)
                border.append((loc[0], loc[1]))
                border_check_list.append(loc[0])
            prev_loc = (loc[0], loc[1])

        for i in range(self.xstart, self.xstop + 1):
            border.append((self.ystop, i))

        convex_perimeter = 0
        for b in border:
            if (b[0] - 1, b[1]) in border:
                convex_perimeter += 1
            if (b[0], b[1] - 1) in border:
                convex_perimeter += 1
            if (b[0] + 1, b[1]) in border:
                convex_perimeter += 1
            if (b[0], b[1] + 1) in border:
                convex_perimeter += 1

        xlen = self.xstop - self.xstart
        ylen = self.ystop - self.ystart
        perimeter = (xlen + ylen) * 2

        self.convexity = perimeter / convex_perimeter

        if ylen > xlen:
            self.ratio = (self.xstop - self.xstart) / (self.ystop - self.ystart)
        else:
            self.ratio = (self.ystop - self.ystart) / (self.xstop - self.xstart)
