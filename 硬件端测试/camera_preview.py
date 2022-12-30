from picamera import PiCamera
from time import sleep
from omxplayer.player import OMXPlayer
from pathlib import Path
from time import sleep
import os
camera = PiCamera()
i = 0
#mp3_1_path = Path("/home/pi/Zood.mp3")
#mp3_2_path = Path("/home/pi/I_got_smoke.mp3")
camera.start_preview()
sleep(10)
while 1:
    camera.capture('/home/pi/Desktop/safely_driving/test/c7/image%s.jpg'% i)
    i= i+ 1
    #os.remove('/home/pi/Desktop/image.jpg'
    #print(i)
camera.stop_preview()