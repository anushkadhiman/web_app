import cv2
import numpy as np
from tensorflow.keras.models import load_model

# defining face detector
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6



class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(1)
       self.model = load_model("model2.h5", compile = False)
    
    def __del__(self):
        #releasing camera
        self.video.release()

    def get_frame(self):
        #extracting frames
        ret, frame = self.video.read()
        print(frame)
        scale_percent = 80 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        #frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)                    
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            im_f=gray[y:y+h,x:x+w]            
            im_f=cv2.resize(im_f,(32,32)) 
            #print(im_f.shape)
            im1 = np.expand_dims(im_f, -1)   
            #print(im1.shape)
            im1 = np.expand_dims(im1, 0)
            #print(im1.shape)
            name=self.model.predict(im1)
            
            result = np.argmax(name)
            #print(result)
            if result == 11:
                face_name = 'Anushka'
            else:
                face_name = 'Unknown'
            
            cv2.putText(frame, face_name, (x,y), cv2.FONT_ITALIC, 1,
                        (255,0,255),2,cv2.LINE_AA)

        # encode OpenCV raw frame to jpg and displaying it
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()