import numpy as np #numpy for converting into array
import cv2 #to convert images
import os #to get the data from the path or to set the path
x=20
y=30
def faceDetection(test_img):
    gray_image=cv2.cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY) #converting image into gray scale because things gone be eassy with gray color rather than color image
    face_haar=cv2.CascadeClassifier(r'C:\Users\saish\OneDrive\Desktop\shashi\facerecongnition\haarcascade_frontalface_alt.xml')
    faces=face_haar.detectMultiScale(gray_image,scaleFactor=1.3,minNeighbors=3)  
    return faces,gray_image

#faceDetection(r'C:\Users\saish\OneDrive\Desktop\shashi\facerecongnition\DSC_0150.jpg')
def labels_for_trainingdata(directory):
    faces=[]
    faceid=[]

    for path,subdirnames,filenames in os.walk(directory):
        #searchers for the file name with other than starthing with i and passes the messeage
        for filename in filenames:
            if filename.startswith("."):
                print("skip the file")
                continue
            id=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print("img_path",img_path)
            print("id",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("not loaded")
                continue
            #detect the faces and make a rectangle
            faces_rect,gray_img=faceDetection(test_img) 
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceid.append(int(id))
    return faces,faceid
#traing vlassifer
def train_classifier(faces,faceid):
    #local binary patterns histograms
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    #face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceid))
    return face_recognizer
# draw rectangle
def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
#put the text
def put_text(test_img,lable_name,x,y):
     cv2.putText(test_img,lable_name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,255,255),3,8)            