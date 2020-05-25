import numpy as np 
import cv2
import os
import facerecognition as fr #import previous file facerecognition

test_img=cv2.imread(r'C:\Users\saish\OneDrive\Desktop\shashi\facerecongnition\IMG_20200224_194247_237.jpg')
faces_detected,gray_image=fr.faceDetection(test_img)

print("face dectected",faces_detected)

faces,faceid=fr.labels_for_trainingdata(r'C:\Users\saish\OneDrive\Desktop\shashi\facerecongnition\myimages')
face_recognizer=fr.train_classifier(faces,faceid)
face_recognizer.save(r'C:\Users\saish\OneDrive\Desktop\shashi\facerecongnition\trainingdata.yml')
name={0:'saishashidhar',1:'sss'}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_image[y:y+w,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print(label)
    print(confidence)
    fr.draw_rect(test_img,face)
    predict_name=name[label]
    fr.put_text(test_img,predict_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
