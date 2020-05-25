import cv2
import sys
c=0

videostream=cv2.VideoCapture(0)

while True:
    ret,frame = videostream.read()
    cv2.imshow('test frame',frame)
    cv2.imwrite(r"C:\Users\saish\OneDrive\Desktop\shashi\facerecongnition\myimages\0\image%04i.jpg" %c,frame)
    c +=1

    if cv2.waitKey(10)==ord('q'):
        break