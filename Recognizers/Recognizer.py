import threading
from abc import ABCMeta, abstractmethod, abstractproperty

class Recognizer(threading.Thread):
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()
        self.setDaemon(True)

    @abstractmethod
    def run(self):
        """Thread function."""
            
    @abstractmethod
    def setNext(self, img_time, img, faces):
        """Set next data for face recognition.
        @param img_time - image capture time
        @param img - numpy array
        @param faces - (x, y, w, h)
        """

    @abstractmethod
    def getLast(self):
        """Return last names.
        @return (img_time, img, faces, names)
        img_time - number
        img - numpy frame
        boxes - (x, y, w, h)
        names - [name0, name1 ...]
        """
