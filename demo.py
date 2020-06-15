from flask import Flask, request, url_for, redirect, render_template, Response, send_from_directory
from camera import VideoCamera
import numpy as np
from cam_agender import VideoCamera_agender
#import pandas as pd
#from skimage import io
import os
 
from keras.preprocessing.image import  load_img, img_to_array
#from werkzeug import SharedDataMiddleware

from keras.models import load_model
#import skimage.transform as st

from werkzeug.utils import secure_filename
#import classification
import time
import uuid
import base64
import requests

from keras.backend import clear_session


clear_session()
# from werkzeug import secure_filename
# import os


app = Flask(__name__, template_folder='template', static_folder='static', static_url_path='/static')

#UPLOAD_FOLDER = 'C:/Users/AD/Downloads/web'


class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#app = Flask(__name__)
#UPLOAD_FOLDER = 'C:/Users/AD/Downloads/web'

UPLOAD_FOLDER = 'uploads'
#ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

#app = Flask(__name__, template_folder='template')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


############################################################################################################


def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file):
    saved_model = 'tuned_model_fin.h5'
    model = load_model(saved_model)
    print('load model!')
    img_width, img_height = 32, 32
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x.transpose(2, 0, 1), axis=0)
    
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    class_name = class_names[answer]
    return answer, class_name

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

'''def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS'''


#####################################################################################################
#################################### Face-Recognition ####################################
############################################################################################################ 

@app.route("/")
def home():
    return render_template("main.html")

@app.route('/')
def template():
    return render_template('index2_1.html')

@app.route("/document1", methods=['GET', 'POST'])
def face_recog():
    # if request.method == 'POST':
        
    #     return redirect(url_for('home'))

    return render_template('index2_1.html')

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
           
@app.route('/video_feed',methods=['GET', 'POST'])
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream/<selected>', methods=['GET', 'POST'])
def stop_stream(selected):
    #selected = request.args.get('type')
    #print(selected) 
    selected = "STOP"
    if (selected == "STOP"):    
        return render_template('index2_2.html', option=selected)   



#####################################################################################################
#################################### Image-Object-Recognition ####################################
############################################################################################################ 

      
@app.route('/')
def template_test():
    return render_template('template.html')


@app.route('/document2', methods=['GET', 'POST'])
def image_clss():
    if request.method == 'POST':
        start_time = time.time()
        file = request.files['file']

        #if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(file_path)
        result = predict(file_path)

        label = result[1]
        
        filename = my_random_string(6) + filename

        os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("--- %s seconds ---" % str (time.time() - start_time))
        
    else: # Option 2: set `labels` to `None` in an `else` statement in case the `if` statement check returns False
        label = None
        filename = ''
        
        
    return render_template('template.html', label=label, imagesource='../uploads/' + filename)
        


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


#####################################################################################################
#################################### Video-Object-Recognition ####################################
############################################################################################################ 
   
@app.route("/document3", methods=['GET', 'POST'])
def video_clss():
    if request.method == 'POST':
        
        return redirect(url_for('home'))
    
    return render_template("vid.html")




#####################################################################################################
#################################### Face-Expression ####################################
############################################################################################################ 



@app.route('/')
def template_last():
    return render_template('face_exp.html')

@app.route("/document4", methods=['GET', 'POST'])
def face_expres():
    if request.method == 'POST':
        
        return redirect(url_for('home'))
    
    return render_template("face_exp.html")


def gen_exp(cam):
    while True:
        #get camera frame
        fr = cam.get_frame_exp()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + fr + b'\r\n\r\n')
        
           
@app.route('/video_feed_exp',methods=['GET', 'POST'])
def video_feed_exp():
    return Response(gen_exp(VideoCamera_agender()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream_exp/<selected>', methods=['GET', 'POST'])
def stop_stream_exp(selected):
    #selected = request.args.get('type')
    #print(selected) 
    selected = "STOP"
    if (selected == "STOP"):    
        return render_template('face_exp_2.html', option=selected)   

    
if __name__ == "__main__":
    app.run(debug=False)
    