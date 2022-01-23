# Dense Descriptor dataset subsampling
# Author : Munch Quentin, 2022.

import math
from random import randint
import numpy as np
import cv2
import matplotlib.pyplot as plt

FileToStore = "/home/neurotronics/Bureau/DDN/dataset/sequence"
VideoFile = "/home/neurotronics/Bureau/DDN/dataset/test.mp4"
sampling = 15
idx = 0
# load video file
cap = cv2.VideoCapture(VideoFile)
counter = 0
# sample every n frame
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if counter % sampling == 0:
        # store frame
        print(FileToStore+'/'+str(idx)+'.png')
        cv2.imwrite(FileToStore+"/"+str(int(idx))+".png", cv2.resize(frame, (640,480)))
        idx+=1
    counter+=1
