from omxplayer.player import OMXPlayer
from pathlib import Path
from time import sleep

mp3_path = Path("/home/pi/Desktop/safely_driving/audio/operating radio.mp3")
while 1 :
    player = OMXPlayer(mp3_path)
    sleep(5)
    #player.stop()
    player.quit()
    print(0)