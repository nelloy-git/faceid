import cv2
import time
import threading

from Trackers.Tracker import Tracker

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

class TrackerOpenCV(Tracker):

    def __init__(self, tracker_type):
        super().__init__()

        self.__lock = threading.Lock()

        self.__creator = OPENCV_OBJECT_TRACKERS[tracker_type]
        self.__tracker = None
        self.__update = False
        self.__update_time = -1
        self.__update_img = None
        self.__update_boxes = None

        self.__next_img_time = None
        self.__next_img = None

        self.__last_img_time = None
        self.__last_img = None
        self.__last_faces = None

    def update(self, img_time, img, boxes):
        """Update tracker data.
        @param img - numpy array
        @param boxes - [(x, y, w, h), ...] """

        self.__lock.acquire()
        if img_time == self.__update_time:
            self.__lock.release()
            return

        self.__update = True
        self.__update_time = img_time
        self.__update_img = img
        self.__update_boxes = boxes
        self.__lock.release()

    def run(self):
        """Thread function."""
        while True:
            self.__lock.acquire()
            update = self.__update
            tracker = self.__tracker
            self.__lock.release()

            if update:
                start_time = time.time()
                tracker = cv2.MultiTracker_create()
                for box in self.__update_boxes:
                    tracker.add(self.__creator(), self.__update_img, box)
                print('Tracker update: ', 1000 * (time.time() - start_time))

            self.__lock.acquire()
            self.__update = False
            self.__tracker = tracker
            self.__lock.release()
            
            if self.__next_img is None or self.__next_img_time == self.__last_img_time or self.__tracker is None:
                time.sleep(0.001)
                continue

            start_time = time.time()
            self.__lock.acquire()
            tracker = self.__tracker

            img_time = self.__next_img_time
            img = self.__next_img
            self.__lock.release()

            (success, cur_faces) = tracker.update(img)

            self.__lock.acquire()
            self.__last_img_time = img_time
            self.__last_img = img
            self.__last_faces = cur_faces
            self.__lock.release()
            print('Tracking: ', 1000 * (time.time() - start_time))

    def setNext(self, img_time, img):
        """Set next data for tracking.
        @param img_time - image capture time
        @param img - numpy array """
        self.__lock.acquire()
        self.__next_img_time = img_time
        self.__next_img = img
        self.__lock.release()

    def getLast(self):
        """Return last tracking data.
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
