import threading
from abc import ABCMeta, abstractmethod, abstractproperty

class Tracker(threading.Thread):
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()
        self.setDaemon(True)

    @abstractmethod
    def update(self, img, boxes):
        """Update tracker data.
        @param img - numpy array
        @param boxes - [(x, y, w, h), ...] """

    @abstractmethod
    def run(self):
        """Thread function."""
            
    @abstractmethod
    def setNext(self, img_time, img):
        """Set next data for tracking.
        @param img_time - image capture time
        @param img - numpy array """

    @abstractmethod
    def getLast(self):
        """Return last tracking data.
        @return (img_time, img, boxes)
        img_time - number
        img - numpy frame
        boxes - (x, y, w, h)
        """
