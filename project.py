import cv2
import numpy as np
import os

frameWidth=640
frameHeight=480

cap=cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)

arr=[]
x=y=a=b=0
ctr=0
while True:
    _,img=cap.read()
    cv2.rectangle(img,(300,300),(100,100),(0,255,0),0)
    crop=img[100:300,100:300]
    imgGray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    #imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    #hsv=cv2.cvtColor(imgBlur,cv2.COLOR_BGR2HSV)
    #mask=cv2.inRange(hsv,np.array([10,23,82]),np.array([179,255,255]))
    
    #res=cv2.bitwise_and(crop,crop,mask=mask)
    #cv2.imshow('res',mask)
    kernel = np.ones((5, 5))
    #imgDil = cv2.dilate(imgBlur, kernel, iterations=1)
    thresh=cv2.threshold(imgGray,120,255,cv2.THRESH_BINARY_INV)[1]
    try:
        cnt=max(cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0], key=lambda x:cv2.contourArea(x))
    
        x,y,a,b=cv2.boundingRect(cnt)
        if(a*b>10000):
            cv2.rectangle(crop,(x,y),(x+a,y+b),(0,0,255),0)
            arr.append(thresh)
            s=os.getcwd()+"/Pics/Six/Six"+str(ctr)+".jpg"
            cv2.imwrite(s,thresh)
            ctr=ctr+1
    except ValueError:
        ctr=ctr
        
    cv2.imshow('sdf',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if(len(arr)>99):
        break


    
    
