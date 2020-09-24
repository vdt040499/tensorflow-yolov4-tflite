from subprocess import *
import time
Popen('python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/VID_plate_3_cut.mp4 --output ./detections/results.avi')
# Popen('python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi')
time.sleep(1)
Popen('python api.py')