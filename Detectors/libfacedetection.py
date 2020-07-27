import ctypes
import numpy as np
import cv2
import pathlib

cur_path = pathlib.Path(__file__).parent.absolute()

clib = ctypes.CDLL(str(cur_path) + '/libfacedetection/build/libfacedetection.so.v0.0.1')

#int * facedetect_cnn(unsigned char * result_buffer, //buffer memory for storing face detection results, !!its size must be 0x20000 Bytes!!
#                    unsigned char * rgb_image_data, int width, int height, int step); //input image, it must be BGR (three channels) insteed of RGB image!

clib.facedetect_cnn.restype = ctypes.POINTER(ctypes.c_int)
clib.facedetect_cnn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

c_str_buffer = ctypes.create_string_buffer(0x20000)
k = 1

def facedetect(img, treshold=90):
    num_rows = img.shape[0]
    num_cols = img.shape[1]
    if len(img.shape) > 2:
        num_chanels = img.shape[2]
    else:
        num_chanels = 1
    found = clib.facedetect_cnn(c_str_buffer, img.ctypes.data_as(ctypes.c_char_p), num_cols, num_rows, num_cols * num_chanels)

    res = []
    count = found[0]
    found = ctypes.cast(found, ctypes.POINTER(ctypes.c_short))
    for i in range(0, count):
        pos = 2 + 142*i
        conf = found[pos]
        x = found[pos + 1] 
        y = found[pos + 2]
        w = found[pos + 3]
        h = found[pos + 4]
        if conf >= treshold:
            res.append((x, y, w, h))  # left
    return res
