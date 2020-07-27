import cv2
import time
import threading
import face_recognition
import pickle
from multiprocessing import Queue

from Detectors.Detector import Detector
import Detectors.libfacedetection as faceboxes

class DetectorLibfacedetection(Detector):

    def __init__(self, pause):
        super().__init__()
        self.__lock = threading.Lock()
        self.__pause = pause

        self.__next_img_time = None
        self.__next_img = None

        self.__last_img_time = None
        self.__last_img = None
        self.__last_faces = None

    def run(self):
        """Thread function."""
        while True:
            if self.__next_img is None or self.__next_img_time == self.__last_img_time:
                time.sleep(0.001)
                continue

            start_time = time.time()
            self.__lock.acquire()
            cur_img_time = self.__next_img_time
            cur_img = self.__next_img
            self.__lock.release()

            cur_boxes = faceboxes.facedetect(cur_img)

            self.__lock.acquire()
            self.__last_img_time = cur_img_time
            self.__last_img = cur_img
            self.__last_faces = cur_boxes
            self.__lock.release()

            print('Detector: ', 1000 * (time.time() - start_time))

            time.sleep(self.__pause)

    def setNext(self, img_time, img):
        """Set next data for face detection.
        @param img_time - image capture time
        @param img - numpy array """

        self.__lock.acquire()
        self.__next_img_time = img_time
        self.__next_img = img
        self.__lock.release()

    def getLast(self):
        """Return last faces.
        @return (img_time, img, boxes)
        img_time - number
        img - numpy frame
        boxes - (x, y, w, h)
        """

        if self.__last_img_time is None:
            return -1, None, None

        self.__lock.acquire()
        res = (self.__last_img_time, self.__last_img, self.__last_faces)
        self.__lock.release()
        return res