from absl import app, logging
import cv2
import numpy as np
import os
import requests
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import time
import json

def unlock_cooler():
    GPIO.setup(11, GPIO.OUT)
    p = GPIO.PWM(11, 50)
    p.start(2.5)
    p.ChangeDutyCycle(7.5)
    time.sleep(1.5)
    p.ChangeDutyCycle(2.5)
    time.sleep(1)
    p.stop()

def main(argv):    
    url = 'https://votan-sparking.herokuapp.com/tickets/createticket'
    while True:
        file = open("temp.txt", "r")
        number = str(file.read())
        if len(number) > 10:
            dataArr = number.split("-")
            if (int(dataArr[1]) >= 5):
                while True:
                    reader = SimpleMFRC522()
                    id, text = reader.read()
                    print('IDcard: ', id)
                    print('User: ', text)
                    infoArr = text.split(' - ')
                    studentId = str(infoArr[0])
                    payload = { 'numplate': dataArr[0], 'userId': studentId }    
                    r = requests.post(url, data=payload)
                    d = json.loads(r.text)
                    successRes = d["success"]
                    messRes = d["message"]
                    if successRes == False:
                        print(messRes)
                        break
                    else:
                        print(str(studentId) + 'TAO VE THANH CONG')
                        unlock_cooler()
                        break
     
if __name__ == '__main__':
    app.run(main)