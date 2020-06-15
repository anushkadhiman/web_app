# -*- coding: utf-8 -*-
"""
Created on Tue May 26 21:04:08 2020

@author: AD
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# defining face detector
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6



model = load_model("model2.h5", compile = False)
model_exp = load_model("model_exp.h5") 
#model_age = load_model("data.h5", compile = False)

target = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

class VideoCamera_agender(object):
    def __init__(self):
       #capturing video
       self.vid = cv2.VideoCapture(1)
       self.model = load_model("model2.h5", compile = False)
       self.model_exp = load_model("model_exp.h5") 
      
    
    def __del__(self):
        #releasing camera
        self.vid.release()

    def get_frame_exp(self):
        #extracting frames
        r, fr = self.vid.read()
        #print(frame)
        scale_percent = 80 # percent of original size
        width = int(fr.shape[1] * scale_percent / 100)
        height = int(fr.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        fr = cv2.resize(fr, dim, interpolation = cv2.INTER_AREA)
        #frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)                    
        gray=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
            cv2.rectangle(fr,(x,y),(x+w,y+h),(0,255,0),2)
            image = gray[y:y + h, x:x + w]
            im_f=cv2.resize(image,(32,32))  
            im1 = np.expand_dims(im_f, -1)
            im1 = np.expand_dims(im1, 0)
            name=self.model.predict(im1)
            result = np.argmax(name)
            if result == 11:
                face_name = 'Anushka'
            else:
                face_name = 'Unknown'
                
            im1 = cv2.resize(image, (48, 48))
            im1 = np.expand_dims(im1, -1)
            im1 = np.expand_dims(im1, 0)
            result1 = self.model_exp.predict(im1)
            expression = target[int(np.argmax(result1))]
             
            cv2.putText(fr, face_name + ' ' + 'is' + ' ' + expression,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (64, 128, 191), 2, cv2.LINE_AA)
            #cv2.putText(fr, face_name + ' ' + 'is' + ' ' + expression + ',' + gender + ',' + age ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (64, 128, 191), 2, cv2.LINE_AA)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(0,255,0),2)
          
        
        r, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
        
