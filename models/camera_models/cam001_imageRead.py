import os
import cv2
import queue
from threading import Thread

# Class for creating a thread for reading image from webcam
class ImageReadModel:
    def __init__(self):
        # os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
        self.exit_flag = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.cap = cv2.VideoCapture(0)
        self.thread = Thread(target=self.capture_frames)

    def startThread(self):
        print("Start Camera reading thread")
        self.thread.start()

    def stopThread(self):
        self.exit_flag = True
        self.thread.join()

    def capture_frames(self):
        while not self.exit_flag and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(frame)

    def getImage(self):
        return self.frame_queue.get(timeout=1)