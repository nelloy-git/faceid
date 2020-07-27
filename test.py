import cv2
import random
import time
import sys
import argparse

from Cameras.CameraOpenCV import CameraOpenCV as Camera
from Detectors.DetectorLibfacedetection import DetectorLibfacedetection as Detector
from Recognizers.RecognizerFaceRecognition import RecognizerFaceRecognition as Recognizer
from Trackers.TrackerOpenCV import TrackerOpenCV as Tracker

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-fw", "--width", type=int, default=640,
    help="camera frame width")
ap.add_argument("-fh", "--height", type=int, default=480,
    help="camera frame height")
ap.add_argument("-p", "--pause", type=float, default=3,
    help="facedetector pause")
ap.add_argument("-t", "--tracker", required=True,
    help="tracker type \"csrt\", \"kcf\", \"boosting\", \"mil\", \"tld\", \"medianflow\", \"mosse\"")
args = vars(ap.parse_args())


buff_size = 1
camera = Camera(buff_size, args['width'], args['height'])
camera.start()

tracker = Tracker("medianflow")
tracker.start()

detector = Detector(args['pause'])
detector.start()

recognizer = Recognizer('./Pickles/encodings_py2.pickle')
recognizer.start()

fps_size = 20
fps_queue = []

prev_frame_time = -1
prev_detect_time = -1
prev_track_time = -1
prev_recognize_time = -1
while True:
    # Get data
    is_new = False
    (cur_time, cur_img) = camera.getLast()
    (cur_detect_time, cur_detect_img, cur_detect_faces) = detector.getLast()
    (cur_track_time, cur_track_img, cur_track_faces) = tracker.getLast()
    (cur_recognize_time, cur_recognize_img, cur_recognize_faces, cur_recognize_names) = recognizer.getLast()

    # Update if new camera frame.
    if cur_time > prev_frame_time:
        is_new = True
        prev_frame_time = cur_time
        tracker.setNext(cur_time, cur_img)
        detector.setNext(cur_time, cur_img)

    # Update if new detetor data.
    if cur_detect_time > prev_detect_time:
        tracker.update(cur_detect_time, cur_detect_img, cur_detect_faces)
        recognizer.setNext(cur_detect_time, cur_detect_img, cur_detect_faces)

    # Sleep if no new image to show
    if not is_new:
        time.sleep(0.001)
        continue
    #if cur_detect_time < 0 or cur_track_time < 0 or cur_detect_time == prev_detect_time or cur_track_time == prev_track_time:

    # Update if new detetor data.
    if cur_detect_time > prev_detect_time:
        prev_detect_time = cur_detect_time

    # Update if new tracker data.
    if cur_track_time > prev_track_time:
        prev_track_time = cur_track_time

    # Show last image
    if cur_detect_time > cur_track_time:
        #img = cur_detect_img
        faces = cur_detect_faces
    else:
        #img = cur_track_img
        faces = cur_track_faces

    img = cur_img

    # Draw faces
    if faces is not None:
        for (x, y, w, h) in faces:
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            color = (0, 255, 0)
            cv2.rectangle(img, p1, p2, color)

    if cur_recognize_names is not None and len(faces) == len(cur_recognize_names):
        for ((x, y, w, h), name) in zip(faces, cur_recognize_names):
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            color = (0, 255, 0)
            cv2.putText(img, name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    fps_queue.insert(0, time.time())
    if len(fps_queue) <= fps_size + 1:
        fps = 0
    else:
        fps_queue.pop()
        fps = fps_size / (fps_queue[0] - fps_queue[fps_size])

    cv2.putText(img, str(int(fps)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('Window', img)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break