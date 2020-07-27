import pickle
import time
import threading
import face_recognition

from Recognizers.Recognizer import Recognizer

class RecognizerFaceRecognition(Recognizer):
    
    def __init__(self, encodings_path):
        super().__init__()

        self.__lock = threading.Lock()
        self.__data = pickle.loads(open(encodings_path, "rb").read())

        self.__next_img_time = None
        self.__next_img = None
        self.__next_faces = None

        self.__last_img_time = None
        self.__last_img = None
        self.__last_faces = None
        self.__last_names = None

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
            cur_faces = self.__next_faces
            self.__lock.release()

            enc_boxes = []
            for (x, y, w, h) in cur_faces:
                enc_boxes.append((y, x + w, y + h, x))
            encodings = face_recognition.face_encodings(cur_img, enc_boxes)
            
            # # initialize the list of names for each face detected
            cur_names = []
            # loop over the facial embeddings
            for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(self.__data["encodings"], encoding)

                name = 'Unknown'
                # check to see if we have found a match
                if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        name = self.__data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number of
                    # votes (note: in the event of an unlikely tie Python will
                    # select first entry in the dictionary)
                    name = max(counts, key=counts.get)

                    #for name in counts:
                    #    print(name, counts[name])
                    
                    #print(name)
                    if counts[name] < 12:
                        name = 'Unknown'
                
                # update the list of names
                cur_names.append(name)

            self.__lock.acquire()
            self.__last_img_time = cur_img_time
            self.__last_img = cur_img
            self.__last_faces = cur_faces
            self.__last_names = cur_names
            self.__lock.release()

            print('Recognizer: ', 1000 * (time.time() - start_time))

    def setNext(self, img_time, img, faces):
        """Set next data for face recognition.
        @param img_time - image capture time
        @param img - numpy array
        @param faces - (x, y, w, h)
        """

        self.__lock.acquire()
        self.__next_img_time = img_time
        self.__next_img = img
        self.__next_faces = faces
        self.__lock.release()

    def getLast(self):
        """Return last names.
        @return (img_time, img, faces, names)
        img_time - number
        img - numpy frame
        boxes - (x, y, w, h)
        names - [name0, name1 ...]
        """

        while self.__last_img_time is None:
            return -1, None, None, None

        self.__lock.acquire()
        res = (self.__last_img_time, self.__last_img, self.__last_faces, self.__last_names)
        self.__lock.release()
        return res
    