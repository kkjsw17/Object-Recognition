import sys
import cv2
from processor import Processor
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QMessageBox


class MyApp(QMainWindow):
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
        videoLoad = QAction('Open', self)
        videoLoad.setShortcut('Ctrl+O')
        videoLoad.setStatusTip('Open Video File')
        videoLoad.triggered.connect(self.loadVideo)

        self.statusBar()

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(videoLoad)

    def loadVideo(self):
        fileName = QFileDialog.getOpenFileName(self, 'Open File', '/', filter='Videos (*.avi *.mov *.mp4)')

        if fileName[0]:
            cap = cv2.VideoCapture(fileName[0])

            if cap.isOpened():
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frameRate = 30

                # u = np.array([152, 106])
                # B = 2.105 * 255

                retval, prev_frame = cap.read()
                if not retval:
                    sys.exit()

                while True:
                    retval, frame = cap.read()
                    if not retval:
                        break

                    prc = Processor(frame)

                    prc.gaussianFiltering((5, 5), 1.2)
                    # prc.detectSkin(0, 35, 58, 173)
                    # prc.connectedComponentLabeling()
                    # prcImage = prc.src
                    flow = prc.motionEstimation(prev_frame)
                    vis_flow = prc.draw_flow(flow)

                    # for i in range(frameHeight):
                    #     for j in range(frameWidth):
                    #         if prcImage[i, j, 0] == 0 and prcImage[i, j, 1] == 0 and prcImage[i, j, 2] == 0:
                    #             frame[i, j] = (0, 0, 0)

                    cv2.imshow('frame', vis_flow)
                    key = cv2.waitKey(frameRate)

                    if key == 27:
                        break

                    prev_frame = frame

                cap.release()
                cv2.destroyAllWindows()
        else:
            QMessageBox.information(self, 'QMessageBox', f"{fileName}이 존재하지 않습니다.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
