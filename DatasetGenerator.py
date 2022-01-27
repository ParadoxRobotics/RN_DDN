# Dense Descriptor dataset subsampling
# Author : Munch Quentin, 2022.
import os
import os.path
import math
from random import randint
import numpy as np
import cv2
import matplotlib.pyplot as plt

FileToStoreImgA = "/home/main/Bureau/dataset/ImgA"
FileToStoreImgB = "/home/main/Bureau/dataset/ImgB"
VideoFile = "/home/main/Bureau/seq1.mp4"
sampling = 10
step = 0
# Check if there are some files in the folder
listFilesA = os.listdir(FileToStoreImgA)
listFilesB = os.listdir(FileToStoreImgB)
if len(listFilesA) > 0:
    idx = len(listFilesA)
    for i in range(0, len(listFilesA)):
        print(listFilesA[i])
        print(listFilesB[i])
else:
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
        if step % 2 == 0:
            # Match A
            print("Img A", FileToStoreImgA+'/'+str(idx)+'.png')
            cv2.imwrite(FileToStoreImgA+"/"+str(int(idx))+".png", cv2.resize(frame, (640,480)))
        else:
            # Match B
            print("Img B", FileToStoreImgB+'/'+str(idx)+'.png')
            cv2.imwrite(FileToStoreImgB+"/"+str(int(idx))+".png", cv2.resize(frame, (640,480)))
            idx+=1
        step+=1
    counter+=1
