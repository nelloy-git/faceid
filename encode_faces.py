# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import faceboxes

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
    help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
    help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
        len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w = image.shape[1]
    h = image.shape[0]
    if w > 1280 or h > 1024:
        kw = 1280 / w
        kh = 1024 / h
        k = min(kw, kh)
        image = cv2.resize(image, (int(k * w), int(k * h)))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image

    boxes = faceboxes.facedetect(image)

    #boxes = face_recognition.face_locations(rgb,
    #    model=args["detection_method"])
    #for box in boxes:
    #    cv2.rectangle(image, (box[3], box[0]), (box[1], box[2]), (255, 0, 0), 3)


    enc_boxes = []
    for (x, y, w, h) in boxes:
        enc_boxes.append((y, x + w, y + h, x))
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
    encodings = face_recognition.face_encodings(rgb, enc_boxes)

    #cv2.imshow('Window', image)
    #key = cv2.waitKey()
    #if key == ord('q'):
    #    break

    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data, protocol=2))
f.close()
