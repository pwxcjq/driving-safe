from keras.models import load_model
import os
import cv2
import glob
import time
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import pandas as pd
from keras.models import *
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from picamera import PiCamera
from omxplayer.player import OMXPlayer
from pathlib import Path
import tensorflow as tf

model_name = 'run_model'
model_path = '/home/pi/Desktop/safely_driving/model/' + model_name + '.h5'
img_path = '/home/pi/Desktop/image.jpg'
last_result = ''
current_result = ''
i = 0

def MakePrediction():
    img = cv2.imread(img_path)
    img = cv2.resize(img,(480,320))
    status = ["safe driving", "texting-right", "phone-right", "texting-left",
		"phone-left", "operating radio", "drinking", "reaching behind",
		"hair and makeup", "talking"]
    x = img.copy()
    x.astype(np.float32)
    predictions = model.predict(np.expand_dims(x, axis=0))
    predictions = predictions[0]
    max_idx = np.argmax(predictions)
    result = str(status[max_idx])
    probability = predictions[max_idx]
    return result, probability

model = load_model(model_path)
#model = tf.keras.models.load_model('Safely_Driving_Model.h5')
camera = PiCamera()
camera.start_preview()
time.sleep(8)

while True:
    camera.capture('/home/pi/Desktop/image.jpg')
    prediction = MakePrediction()
    current_result = prediction[0]
    probability = prediction[1]
    print('current_result is: ' + str(current_result))
    #print('last_result is: ' + str(last_result))
    print('probability is: ' + str(probability))
    if current_result != 'safe driving' and last_result != current_result:
        audio_path = '/home/pi/Desktop/safely_driving/audio/' + str(current_result) + '.mp3'
        print(audio_path)
        player = OMXPlayer(Path(audio_path))
        time.sleep(5)
        #player.stop()
        player.quit()
    os.remove(img_path)
    last_result = current_result
    time.sleep(1.5)
camera.stop_preview()