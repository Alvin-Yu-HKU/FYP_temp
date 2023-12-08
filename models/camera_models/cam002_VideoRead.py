import os
import cv2
import queue
from threading import Thread

# Class for creating a thread for reading image from video
class VideoReadModel:
    def __init__(self, video_path):
        # os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
        self.exit_flag = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.video = cv2.VideoCapture(video_path)       
        if not self.video.isOpened():
            print("Error opening video file")

    def startThread(self):
        print("Dummy Video Reading thread start")


    def stopThread(self):
        print("Dummy Video Reading thread end")

    def getImage(self):
        ret, frame = self.video.read()
        if not ret:
            print("Reset")
            self.video.release()
            self.video = cv2.VideoCapture('test.mp4')
            ret, frame = self.video.read()
            return frame
        return frame