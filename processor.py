import numpy as np
from math import pi
import cv2
import collections


class Processor:
    def __init__(self, src):
        self.src = src
        self.Height = src.shape[0]
        self.Width = src.shape[1]
        self.Channel = src.shape[2]

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

    def motionEstimation(self, prev):
        now_frame = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(now_frame, prev_frame, None, 0.5, 3, 13, 3, 5, 1.1, 0)

        return flow

    def draw_flow(self, flow, step=16):
        gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        # 입력 영상의 컬러 영상 변환
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 직선 그리기
        cv2.polylines(vis, lines, 0, (0, 255, 255), lineType=cv2.LINE_AA)

        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 128, 255), -1, lineType=cv2.LINE_AA)

        return vis

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

        kernel = np.ones((5, 5), dtype='uint8')
        morp = cv2.dilate(cv2.erode(otsu, kernel, iterations=1), kernel, iterations=1)

        pad_img = np.zeros((self.Height + 2, self.Width + 2), dtype='uint8')
        pad_img[1:1 + self.Height, 1:1 + self.Width] = morp
        comp = np.zeros((self.Height + 2, self.Width + 2))
        cmpcnt = 0
        eqvTable = {}
        cmplist = []
        color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                      (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
                      (128, 0, 128), (0, 128, 128)]

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
                    if comp[i, j - 1] != 0 and not flag:
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

        orderedEqvTable = collections.OrderedDict(sorted(eqvTable.items(), reverse=True))

        for key in orderedEqvTable.keys():
            for i in range(self.Height + 2):
                for j in range(self.Width + 2):
                    if comp[i, j] == key:
                        comp[i, j] = eqvTable[key]

        for i in range(self.Height + 2):
            for j in range(self.Width + 2):
                if comp[i, j] != 0 and cmplist.count(comp[i, j]) == 0:
                    cmplist.append(comp[i, j])

        ret_img = np.zeros((self.Height, self.Width, 3), dtype='uint8')

        for i in range(self.Height):
            for j in range(self.Width):
                cnt = 0
                for cmpnum in cmplist[0:len(color_list)]:
                    if comp[i + 1, j + 1] == cmpnum:
                        ret_img[i, j] = color_list[cnt]
                    cnt += 1

        self.src = ret_img
