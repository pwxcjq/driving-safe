# coding: utf-8
# last modified:2022-11-27
# last coder:李俊懿
import time
import serial
import re
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

def Convert_to_degrees(in_data1, in_data2):
    len_data1 = len(in_data1)
    str_data2 = "%05d" % int(in_data2)
    temp_data = int(in_data1)
    symbol = 1
    if temp_data < 0:
        symbol = -1
    degree = int(temp_data / 100.0)
    str_decimal = str(in_data1[len_data1-2]) + str(in_data1[len_data1-1]) + str(str_data2)
    f_degree = int(str_decimal)/60.0/100000.0
    # print("f_degree:", f_degree)
    if symbol > 0:
        result = degree + f_degree
    else:
        result = degree - f_degree
    return result

def GPS_read():
        global utctime
        global lat
        global ulat
        global lon
        global ulon
        global numSv
        global msl
        global cogt
        global cogm
        global sog
        global kph
        global gps_t
        if ser.inWaiting():
            if ser.read(1) == b'G':
                if ser.inWaiting():
                    if ser.read(1) == b'N':
                        if ser.inWaiting():
                            choice = ser.read(1)
                            if choice == b'G':
                                if ser.inWaiting():
                                    if ser.read(1) == b'G':
                                        if ser.inWaiting():
                                            if ser.read(1) == b'A':
                                                #utctime = ser.read(7)
                                                GGA = ser.read(70)
                                                GGA_g = re.findall(r"\w+(?=,)|(?<=,)\w+", str(GGA))
                                                # print(GGA_g)
                                                if len(GGA_g) < 13:
                                                    print("GPS no found")
                                                    gps_t = 0
                                                    return 0
                                                else:
                                                    utctime = GGA_g[0]
                                                    # lat = GGA_g[2][0]+GGA_g[2][1]+'°'+GGA_g[2][2]+GGA_g[2][3]+'.'+GGA_g[3]+'\''
                                                    lat = "%.8f" % Convert_to_degrees(str(GGA_g[2]), str(GGA_g[3]))
                                                    ulat = GGA_g[4]
                                                    # lon = GGA_g[5][0]+GGA_g[5][1]+GGA_g[5][2]+'°'+GGA_g[5][3]+GGA_g[5][4]+'.'+GGA_g[6]+'\''
                                                    lon = "%.8f" % Convert_to_degrees(str(GGA_g[5]), str(GGA_g[6]))
                                                    ulon = GGA_g[7]
                                                    numSv = GGA_g[9]
                                                    msl = GGA_g[12]+'.'+GGA_g[13]+GGA_g[14]
                                                    #print(GGA_g)
                                                    gps_t = 1
                                                    return 1
                            elif choice == b'V':
                                if ser.inWaiting():
                                    if ser.read(1) == b'T':
                                        if ser.inWaiting():
                                            if ser.read(1) == b'G':
                                                if gps_t == 1:
                                                    VTG = ser.read(40)
                                                    VTG_g = re.findall(r"\w+(?=,)|(?<=,)\w+", str(VTG))
                                                    cogt = VTG_g[0]+'.'+VTG_g[1]+'T'
                                                    if VTG_g[3] == 'M':
                                                        cogm = '0.00'
                                                        sog = VTG_g[4]+'.'+VTG_g[5]
                                                        kph = VTG_g[7]+'.'+VTG_g[8]
                                                    elif VTG_g[3] != 'M':
                                                        cogm = VTG_g[3]+'.'+VTG_g[4]
                                                        sog = VTG_g[6]+'.'+VTG_g[7]
                                                        kph = VTG_g[9]+'.'+VTG_g[10]
                                                #print(kph)

model_name = 'run_model'
model_path = '/home/pi/Desktop/safely_driving/model/' + model_name + '.h5'
img_path = '/home/pi/Desktop/img.jpg'
last_result = ''
current_result = ''
model = load_model(model_path)
#model = tf.keras.models.load_model('Safely_Driving_Model.h5')
camera = PiCamera()
utctime = ''
lat = ''
ulat = ''
lon = ''
ulon = ''
numSv = ''
msl = ''
cogt = ''
cogm = ''
sog = ''
kph = '0'
gps_t = 0
ser = serial.Serial("/dev/ttyUSB0", 9600)
camera.start_preview()

if ser.isOpen():
    print("GPS Serial Opened! Baudrate=9600")
else:
    print("GPS Serial Open Failed!")

try:
    while True:
        if GPS_read():

            print("*********************")
            print('UTC Time:'+utctime)
            print('纬度:'+lat+ulat)
            print('经度:'+lon+ulon)
            #print('Number of satellites:'+numSv)
            print('海拔高度:'+msl)
            print('方位角:'+cogt+'°')
            print('磁航向:'+cogm+'°')
            #print('Ground speed:'+sog+'Kn')
            print('行驶速度'+kph+'Km/h')
            print("*********************")
            speed = float(kph)
            if speed > 1:
                camera.capture('/home/pi/Desktop/img.jpg')
                prediction = MakePrediction()
                current_result = prediction[0]
                probability = prediction[1]
                print('current_result is: ' + str(current_result))
                print('last_result is: ' + str(last_result))
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
                time.sleep(1)


except KeyboardInterrupt:
    ser.close()
    print("GPS serial Close!")

camera.stop_preview()