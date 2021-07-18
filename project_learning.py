import tensorflow as tf
import os
import cv2
import numpy as np
from gtts import gTTS
labels=np.array([], dtype=np.int8)
arr=[]
ctr=-1
i=0
s=""
st=""
for j in range (0,7):
    if(j==0):
        s="Zero/Zero"
    elif(j==1):
        s="One/One"
    elif (j==2):
        s="Two/Two"
    elif (j==3):
        s="Three/Three"
    elif (j==4):
        s="Four/Four"
    elif(j==5):
        s="Five/Five"
    elif(j==6):
        s="Six/Six"
        
    for i in range(0,100):
        st=os.getcwd()+"/Pics/"+s+str(i)+".jpg"
        A=cv2.imread(st,0)
        
        A=A/255
        arr.append(A)
        labels=np.append(labels,j)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(200, 200)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7)
])

        
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(np.array(arr), labels, epochs=5)


frameWidth=640
frameHeight=480

cap=cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)

arr=[]
x=y=a=b=0
ctr=0
arr1=[]
while True:
    _,img=cap.read()
    cv2.rectangle(img,(300,300),(100,100),(0,255,0),0)
    crop=img[100:300,100:300]
    imgGray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgGray, kernel, iterations=1)
    thresh=cv2.threshold(imgBlur,127,255,cv2.THRESH_BINARY_INV)[1]
    try:
        cnt=max(cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0], key=lambda x:cv2.contourArea(x))
       
        x,y,a,b=cv2.boundingRect(cnt)
        if(a*b>10000):
            cv2.rectangle(crop,(x,y),(x+a,y+b),(0,0,255),0)
            probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
            arr1.append(thresh)
            predictions = probability_model.predict(np.array(arr1))
            arr1=[]
            pred=np.argmax(predictions[0])
            cv2.putText(img,"PREDICTION: "+str(pred),(x+2,y-2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
                        
    except ValueError:
        ctr=ctr
        
    cv2.imshow('sdf',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
text='DONE!'
lang='en'
obj=gTTS(text,lang,slow=False)
obj.save('done.mp3')
os.system('done.mp3')
