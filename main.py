import sys
import cv2
import numpy as np
from processor import Processor
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QMessageBox


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.initMenuBar()

        self.setWindowTitle("Object Recognition Project")
        self.move(0, 0)
        self.resize(800, 400)
        self.show()

    def initMenuBar(self):
        detectSkinAction = QAction('Skin Detection', self)
        detectSkinAction.setShortcut('Ctrl+S')
        detectSkinAction.setStatusTip('Load Video for Detect Skin')
        detectSkinAction.triggered.connect(self.detectSkin)

        detectVehicleAction = QAction('Vehicle Detection', self)
        detectVehicleAction.setShortcut('Ctrl+V')
        detectVehicleAction.setStatusTip('Load Video for Detect Vehicle')
        detectVehicleAction.triggered.connect(self.detectVehicle)

        self.statusBar()

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        menubar.addAction(detectSkinAction)
        menubar.addAction(detectVehicleAction)

    def detectSkin(self):
        fileName = QFileDialog.getOpenFileName(self, 'Open File', '/', filter='Videos (*.avi *.mov *.mp4)')

        if fileName[0]:
            cap = cv2.VideoCapture(fileName[0])

            if cap.isOpened():
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frameRate = 10

                # u = np.array([152, 106])
                # B = 2.105 * 255

                retval, prev_frame = cap.read()
                if not retval:
                    sys.exit()

                while True:
                    retval, frame = cap.read()
                    if not retval:
                        break

                    # frame = cv2.GaussianBlur(frame, (0, 0), 1.2)
                    filtered_frame = cv2.bilateralFilter(frame, 10, 50, 50)
                    prc = Processor(filtered_frame)

                    prc.histEqulization()
                    prc.detectSkin(0, 35, 58, 173)
                    prc.connectedComponentLabeling()
                    prcImage = prc.src

                    for i in range(frameHeight):
                        for j in range(frameWidth):
                            if prcImage[i, j, 0] == 0 and prcImage[i, j, 1] == 0 and prcImage[i, j, 2] == 255:
                                frame[i, j] = (0, 0, 255)

                    cv2.imshow('frame', frame)
                    key = cv2.waitKey(frameRate)

                    if key == 27:
                        break

                    # prev_frame = frame

                cap.release()
                cv2.destroyAllWindows()
        else:
            QMessageBox.information(self, 'QMessageBox', f"{fileName}이 존재하지 않습니다.")

    def detectVehicle(self):
        fileName = QFileDialog.getOpenFileName(self, 'Open File', '/', filter='Videos (*.avi *.mov *.mp4)')

        if fileName[0]:
            cap = cv2.VideoCapture(fileName[0])

            if cap.isOpened():
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                BG = np.zeros((frameHeight, frameWidth, 3))
                cnt = 0

                while cnt < 1000:
                    retval, frame = cap.read()
                    if not retval:
                        break

                    frame = cv2.GaussianBlur(frame, (0, 0), 1.2)
                    BG += frame
                    cnt += 1

                BG = (BG / cnt).astype('uint8')
                cap.release()
                cv2.imshow('background image', BG)

            cap = cv2.VideoCapture(fileName[0])

            if cap.isOpened():
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frameRate = 30
                tmpBG = np.zeros((frameHeight, frameWidth, 3))
                retval, prev_frame = cap.read()
                prev_frame = cv2.GaussianBlur(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
                cnt = 0

                while True:
                    retval, frame = cap.read()
                    if not retval:
                        break

                    filtered_frame = cv2.GaussianBlur(frame, (0, 0), 1.2)

                    cnt += 1
                    tmpBG += filtered_frame
                    if cnt == 1000:
                        BG = (tmpBG / cnt).astype('uint8')
                        cnt = 0
                        tmpBG = np.zeros((frameHeight, frameWidth, 3))

                    prc = Processor(filtered_frame)

                    sub = (prc.src.astype('int64') - BG.astype('int64'))
                    sub[sub < 0] = 0
                    sub = sub.astype('uint8')
                    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
                    ret, sub = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                    moving = (sub.astype('int64') - prev_frame.astype('int64'))
                    moving[moving < 0] = 0
                    moving = moving.astype('uint8')

                    moving = cv2.cvtColor(moving, cv2.COLOR_GRAY2BGR)
                    prc = Processor(moving)
                    prc.connectedComponentLabeling_vehicle()

                    for i in range(frameHeight):
                        for j in range(frameWidth):
                            if prc.src[i, j, 0] == 0 and prc.src[i, j, 1] == 0 and prc.src[i, j, 2] == 255:
                                frame[i, j] = (0, 0, 255)

                    cv2.imshow('frame', frame)
                    key = cv2.waitKey(frameRate)

                    prev_frame = sub

                    if key == 27:
                        break

                cap.release()
                cv2.destroyAllWindows()
        else:
            QMessageBox.information(self, 'QMessageBox', f"{fileName}이 존재하지 않습니다.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())
