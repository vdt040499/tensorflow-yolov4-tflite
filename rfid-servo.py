import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import re
import time

studentId =  ''

def read_RFID():
    global studentId
    reader = SimpleMFRC522()
    id, text = reader.read()
    print(id)
    print(text)
    infoArr = text.split(' - ')
    studentId = str(infoArr[0])
    #GPIO.cleanup()

def unlock_cooler():
    GPIO.setup(11, GPIO.OUT)
    p = GPIO.PWM(11, 50)
    p.start(2.5)
    p.ChangeDutyCycle(7.5)
    time.sleep(1.5)
    p.ChangeDutyCycle(2.5)
    time.sleep(1)
    p.stop()
    #GPIO.cleanup()

read_RFID()
print('studentId: ', studentId)
if studentId == '17520814':
    unlock_cooler()

GPIO.cleanup()
