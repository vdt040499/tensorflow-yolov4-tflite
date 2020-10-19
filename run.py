from subprocess import Popen

# Command to run on Raspberry Pi
# args1 = ['python3', 'detect_video.py', '--weights', './checkpoints/custom-tiny-416.tflite', '--size', '416', '--model', 'yolov4', '--video', './data/video/VID_plate_3_cut.mp4', '--output', './detections/results.avi', '--framework', 'tflite']
# Popen(args1)
# Popen('python detect_video.py --weights ./checkpoints/custom-tiny-416.tflite --size 416 --model yolov4 --video 0 --output ./detections/results.avi --framework tflite')

# args2 = ['python3', 'api.py']
# Popen(args2)

#Command to run on Win
# Popen('python detect_video.py --weights ./checkpoints/custom-tiny-416.tflite --size 416 --model yolov4 --video ./data/video/VID_plate_3_cut.mp4 --output ./detections/results.avi --framework tflite')
Popen('python detect_video.py --weights ./checkpoints/custom-tiny-416.tflite --size 416 --model yolov4 --video 0 --output ./detections/results.avi --framework tflite')

# Popen('python api.py')