import sys
import threading
import time

import cv2
import numpy as np
from numpy import expand_dims
from tensorflow.keras.models import load_model
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from retinaface import RetinaFace

from GUI import Ui_Dialog


def camera_list_ports():
    is_working = True
    dev_port = 0
    working_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        is_reading, img = camera.read()
        if is_reading:
            working_ports.append(str(dev_port))
            camera.release()
        else:
            break
        dev_port += 1
    return working_ports


# Thread for the stream Capture
class Thread(QtCore.QThread):
    any_signal = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(Thread, self).__init__(parent)
        self.stop_thread = False

    def run(self):
        while True:
            time.sleep(0.1)
            if self.stop_thread:
                break
            self.any_signal.emit(0)

    def stop(self):
        self.stop_thread = False


class Main:
    def __init__(self):

        self.current_frame = None
        self.current_face = None
        self.camera_stream = False
        self.current_emotion = ""

        # Load the model
        self.model = load_model("model.h5", compile=False)

        # Load the labels
        self.class_names = ["Angry", "Happy", "Sad", "Surprised"]

        # Load the face detector model
        self.detector = RetinaFace(quality='speed')

        self.Dialog = QtWidgets.QDialog()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self.Dialog)
        self.Dialog.setWindowFlag(Qt.FramelessWindowHint)
        self.Dialog.setAttribute(Qt.WA_TranslucentBackground)

        # Refresh the connected camera list by default
        self.refresh_connected_cameras()

        # Start the gui variable updater thread
        self.thread = Thread(parent=None)
        self.thread.any_signal.connect(self.gui_variable_updater)
        self.thread.start()

        # PushButton connections
        self.ui.pushButton_2.clicked.connect(lambda: sys.exit(0))
        self.ui.pushButton_4.clicked.connect(self.stop_everything_and_close)
        self.ui.pushButton_3.clicked.connect(lambda: threading.Thread(target=self.start_camera_stream).start())

    def gui_variable_updater(self):
        if self.current_frame is not None:
            self.set_image(self.current_frame)
        else:
            self.ui.label_5.clear()

        if self.current_face is not None:
            self.set_face_image(self.current_face)
        else:
            self.ui.label_4.clear()

        self.ui.label_7.setText(f"Prediction: {self.current_emotion}")

    def refresh_connected_cameras(self):
        self.ui.comboBox.clear()
        self.camera_list = camera_list_ports()
        self.ui.comboBox.addItems(self.camera_list)

    def set_image(self, img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = result.shape
        step = channel * width
        qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
        self.ui.label_5.setPixmap(QtGui.QPixmap(qImg))

    def set_face_image(self, img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = result.shape
        step = channel * width
        qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
        self.ui.label_4.setPixmap(QtGui.QPixmap(qImg))

    def stop_everything_and_close(self):
        if not self.camera_stream:
            sys.exit(0)

    def start_camera_stream(self):

        # Check for the camera is working or stop
        if self.ui.pushButton_3.text() == "Start Stream":

            # Change the camera text
            self.ui.pushButton_3.setText("Stop Camera")
            cam_id = int(self.ui.comboBox.currentText())
            self.camera_stream = True
            video = cv2.VideoCapture(cam_id)
            while video.isOpened():

                # Check for the camera stream is working or not
                if not self.camera_stream:
                    break

                # Get the frame from the video
                ret, frame = video.read()

                # Check for the camera is captured or not
                if not ret:
                    break
                else:

                    # Detect faces using RetinaFace
                    detected_faces = self.detector.predict(frame)

                    if len(detected_faces) == 0:
                        self.current_face = None
                        self.current_emotion = ""
                        self.current_frame = frame.copy()

                    else:
                        # Get the face coordinates
                        x1, y1, x2, y2 = detected_faces[0]['x1'], detected_faces[0]['y1'], detected_faces[0]['x2'], \
                            detected_faces[0]['y2']

                        # Draw bounding box around the face
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        emotion = ""

                        try:
                            gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

                            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
                            gray = cv2.resize(gray, (48, 48))
                            gray = gray / 255.

                            emotion = list(self.model.predict(expand_dims(gray, axis=0)))
                            confidence = max(emotion[0])
                            idx = list(emotion[0]).index(confidence)
                            emotion = self.class_names[idx]

                            cv2.putText(frame,
                                        f"{emotion} - {int(round(confidence, 2) * 100)}",
                                        (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                        except:
                            pass

                        # Store the frame
                        self.current_frame = frame.copy()
                        self.current_emotion = emotion
                        self.current_face = frame[y1:y2, x1:x2]

            video.release()
            self.ui.pushButton_3.setText("Start Stream")
            self.current_frame = None
            self.current_face = None
            self.current_emotion = ""
            self.camera_stream = False
            self.ui.label_5.clear()
        else:
            self.ui.pushButton_3.setText("Start Stream")
            self.current_frame = None
            self.current_face = None
            self.current_emotion = ""
            self.camera_stream = False
            self.ui.label_5.clear()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.Dialog.show()
    sys.exit(app.exec_())
