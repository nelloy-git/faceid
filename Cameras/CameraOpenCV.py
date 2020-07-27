import cv2
import time
import threading

from Cameras.Camera import Camera

class CameraOpenCV(Camera):

    def __init__(self, buffer_size, cam_width, cam_height):
        super().__init__()

        for i in range(0, 10):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            time.sleep(0.1)
            if cap.isOpened():
                break
            
        if not cap.isOpened():
            print('Can not open camera.')
            sys.exit(-1)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width);
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height);

        self.__cap = cap
        _, img = cap.read()
        self.__queue = [(time.time(), img)]
        self.__lock = threading.Lock()
        self.__buffer_size = buffer_size

    def run(self):
        """Thread function."""

        while True:
            start_time = time.time()
            _, img = self.__cap.read()

            self.__lock.acquire()
            self.__queue.insert(0, (time.time(), img))
            if len(self.__queue) > self.__buffer_size:
                self.__queue.pop()
            self.__lock.release()

            print('CameraOpenCV: ', 1000 * (time.time() - start_time))

    def getLast(self):
        """Returns the last frame.
        @return (img_time, img)"""
        
        self.__lock.acquire()
        res = self.__queue[0]
        self.__lock.release()

        return res

    def getNext(self, t):
        """Returns the next frame after selected time.
        @param t - time """

        res = self.__queue[0]
        self.__lock.acquire()
        for i in range(len(self.__queue) - 1, -1, -1):
            (img_t, img) = self.__queue[i]
            if img_t > t:
                res = self.__queue[i]
                break
        self.__lock.release()

        return res