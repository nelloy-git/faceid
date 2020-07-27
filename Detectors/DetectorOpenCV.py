import cv2
import time
import threading
import pickle
import numpy as np
from multiprocessing import Queue

from Detectors.Detector import Detector

class DetectorOpenCV(Detector):

    def __init__(self, proto_path, model_path, model_width, model_height, threshold = 0.6):
        super().__init__()
        self.__lock = threading.Lock()
        self.__dnn = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        self.__dnn_size = (model_width, model_height)
        self.__dnn_threshold = threshold

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

            (h, w) = cur_img.shape[:2]
            cur_img = cv2.resize(cur_img, self.__dnn_size)
            blob = cv2.dnn.blobFromImage(cur_img, 1.0, self.__dnn_size,
                                        (104.0, 177.0, 123.0), swapRB=False, crop=False)
            self.__dnn.setInput(blob)
            detections = self.__dnn.forward()

            cur_boxes = []
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > self.__dnn_threshold:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    cur_boxes.append((startX, startY, endX - startX, endY - startY))
            
            self.__lock.acquire()
            self.__last_img_time = cur_img_time
            self.__last_img = cur_img
            self.__last_faces = cur_boxes
            self.__lock.release()

            print('Detector: ', 1000 * (time.time() - start_time))

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