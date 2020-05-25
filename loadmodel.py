import numpy as np
import cv2
import os

import facerecognition as fr
print (fr)

test_img=cv2.imread(r'C:\Users\saish\OneDrive\Desktop\shashi\facerecongnition\IMG_20200224_194247_237.jpg')      #Give path to the image which you want to test


faces_detected,gray_img=fr.faceDetection(test_img)
print("face Detected: ",faces_detected)


face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\saish\OneDrive\Desktop\shashi\facerecongnition\trainingdata.yml')  #Give path of where trainingData.yml is saved

name={0:"saishashidhar",1:"Vijay Deverakonda"}             #Change names accordingly.  If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1. 

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print ("Confidence :",confidence)
    print("label :",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>60):
        fr.put_text(test_img,'unkonown',x,y)
        continue
    
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows