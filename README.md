# Real-time-Drowsiness-Detection-System-using-Facial-Landmarks

This project aims to create a Drowsiness Detection System using Convolutional Neural Networks (CNNs) to monitor and alert individuals about their drowsy state during activities such as driving. The system utilizes facial feature analysis, particularly focusing on the eyes, to detect signs of drowsiness.

## Features

- Real-time monitoring of facial features.
- Eye state classification as "Open" or "Closed" using a trained CNN model.
- Continuous tracking of the user's drowsiness score.
- Alarm activation when the drowsiness score surpasses a predefined threshold.
- Simple and intuitive user interface.

## Requirements

- Python
- OpenCV
- Keras
- TensorFlow
- Matplotlib
- Pygame
## Architecture Diagram/Flow

![image](https://github.com/Karthikeyan21001828/Real-time-Drowsiness-Detection-System-using-Facial-Landmarks/assets/93427303/7b211d1a-ebb1-4422-8086-6e1cb291356e)


## Installation

1. Clone the repository: git clone https://github.com/Karthikeyan21001828/Real-time-Drowsiness-Detection-System-using-Facial-Landmarks.git

2. Install dependencies: pip install -r requirements.txt

3. Download the pre-trained model file (cnnCat2.h5) or train a new model using the provided dataset.

4. Ensure the required Haar cascade files (haarcascade_frontalface_alt.xml, haarcascade_lefteye_2splits.xml, haarcascade_righteye_2splits.xml) are available in the specified paths.

5. Run the Drowsiness Detection.py script: python "Drowsiness Detection.py"

## Usage

1. Execute the script, and the webcam feed will open.

2. The system will continuously monitor eye states and update the drowsiness score.

3. A visual and audible alarm will be triggered if the drowsiness score exceeds the threshold.

4. Close the application by pressing 'q' on the keyboard.

## Program:
### model.py
```python
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model
def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):
    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)
BS= 32
TS=(24,24)
train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)
# img,labels= next(train_batch)
# print(img.shape)
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#32 convolution filters used each of size 3x3
#again
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#64 convolution filters used each of size 3x3
#choose the best features via pooling    
#randomly turn neurons on and off to improve convergence
    Dropout(0.25),
#flatten since too many dimensions, we only want a classification output
    Flatten(),
#fully connected to get all relevant data
    Dense(128, activation='relu'),
#one more dropout for convergence' sake :) 
    Dropout(0.5),
#output a softmax to squash the matrix into output probabilities
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)
model.save('models/cnnCat2.h5', overwrite=True)
```
### Drowsiness Detection.py
```python
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
mixer.init()
sound = mixer.Sound('alarm.wav')
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
lbl=['Close','Open']
model = load_model('models/cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict(r_eye)
        rpred = np.argmax(rpred, axis=1)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break
    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict(l_eye)
        lpred = np.argmax(lpred, axis=1)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()      
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
## Output:
![image](https://github.com/Karthikeyan21001828/Real-time-Drowsiness-Detection-System-using-Facial-Landmarks/assets/93427303/2186f245-1093-4bca-910e-32c6c8721972)

![image](https://github.com/Karthikeyan21001828/Real-time-Drowsiness-Detection-System-using-Facial-Landmarks/assets/93427303/ad38c1be-f7dd-4782-bc4f-22881f74e60c)

## Result:

The system provides real-time monitoring of drowsiness levels, alerting users when they show signs of fatigue, especially in situations like driving where staying alert is crucial for safety. The visual and audible alarms aim to prompt users to take breaks or corrective actions to prevent potential accidents due to drowsiness.
