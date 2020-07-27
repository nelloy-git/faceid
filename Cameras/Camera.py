import threading
from abc import ABCMeta, abstractmethod, abstractproperty

class Camera(threading.Thread):
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()
        self.setDaemon(True)

    @abstractmethod
    def run(self):
        """Thread function."""

    @abstractmethod
    def getLast(self):
        """Returns the last frame.
        @return (img_time, img)"""
            
    @abstractmethod
    def getNext(self, t):
        """Returns the next frame after selected time.
        @param t - time """
