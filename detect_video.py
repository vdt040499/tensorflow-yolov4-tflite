import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import plateprocess.handlehalfofplate as handlehalfofplate
from flask import Flask, request, Response, jsonify, send_from_directory, abort

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

try:
    os.mkfifo("temp.txt")
except:
    pass

def ratioCheck(area, width, height):
        min = 10000
        max = 40000

        ratioMin = 2
        ratioMax = 3
        ratio = float(width)/float(height)
        if ratio < 1:
            ratio = 1/ratio
        
        print(area, ratio)
        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True

def crop_rotated_contour(plate, rect):
        """
        Rotate the plate and crop the plate with its rotation
        """
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        W = rect[1][0]
        H = rect[1][1]
        
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        
        angle = rect[2]
        if angle < (-45):
            angle += 90
            
        # Center of rectangle in source image
        center = ((x1 + x2)/2,(y1 + y2)/2)

        # Size of the upright rectangle bounding the rotated rectangle
        size = (x2-x1, y2-y1)
        M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

        # Cropped upright rectangle
        cropped = cv2.getRectSubPix(plate, size, center)
        cropped = cv2.warpAffine(cropped, M, size)
        croppedW = H if H > W else W
        croppedH = H if H < W else W

        # Final cropped & rotated rectangle
        croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0]/2, size[1]/2))
        return croppedRotated


def clean_plate(plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours,h = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas) # index of the largest contour in the area array
            
            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x,y,w,h = cv2.boundingRect(max_cnt)
            rect = cv2.minAreaRect(max_cnt)
            rotatedPlate = crop_rotated_contour(plate, rect)
            if not ratioCheck(max_cntArea, rotatedPlate.shape[1], rotatedPlate.shape[0]):
                return plate, False, None
            return rotatedPlate, True, [x, y, w, h]
        else:
            return plate, False, None

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    plateNumberDict = {}

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image, x, y, w, h = utils.draw_bbox(frame, pred_bbox)

        #Handle license plate
        if x == 0 and y == 0 and w == 0 and h == 0:
            image = cv2.putText(image, "no plate", (0, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        else:
            cropped = frame[int(y):int(y+h), int(x):int(x+w)]
            plate_upper = cropped[0:int(cropped.shape[0]/2), 0:int(cropped.shape[1])]
            clean_upper, plateFound, coordinates = clean_plate(plate_upper)
            plate_lower = cropped[int(cropped.shape[0]/2): int(cropped.shape[0]), 0:int(cropped.shape[1])]
            clean_lower, plateFound, coordinates = clean_plate(plate_lower)
            upper_text = handlehalfofplate.handle(clean_upper)
            print ("Upper_Text = " + upper_text)
            lower_text = handlehalfofplate.handle(clean_lower)
            print ("Lower_text = " + lower_text)
            
            # check length of number plate line 
            if len(upper_text) == 4 and (len(lower_text) == 4 or len(lower_text) == 5):
                # check char at index 2 is character
                if ord(upper_text[2]) > 64 and ord(upper_text[2]) < 91:
                    if (ord(upper_text[0]) > 47 and ord(upper_text[0]) < 58) and (ord(upper_text[1]) > 47 and ord(upper_text[1]) < 58) and (ord(upper_text[3]) > 47 and ord(upper_text[3]) < 58):
                        if len(lower_text) == 4:
                            print("Bằng 4")
                            if (ord(lower_text[0]) > 47 and ord(lower_text[0]) < 58) and (ord(lower_text[1]) > 47 and ord(lower_text[1]) < 58) and (ord(lower_text[2]) > 47 and ord(lower_text[2]) < 58) and (ord(lower_text[3]) > 47 and ord(lower_text[3]) < 58):
                                number_plate = upper_text + " " + lower_text
                                if number_plate in plateNumberDict.keys():
                                    plateNumberDict[str(number_plate)] += 1
                                else:
                                    plateNumberDict[str(number_plate)] = 1
                        if len(lower_text) == 5:
                            print("Bằng 5")
                            if (ord(lower_text[0]) > 47 and ord(lower_text[0]) < 58) and (ord(lower_text[1]) > 47 and ord(lower_text[1]) < 58) and (ord(lower_text[2]) > 47 and ord(lower_text[2]) < 58) and (ord(lower_text[3]) > 47 and ord(lower_text[3]) < 58) and (ord(lower_text[4]) > 47 and ord(lower_text[4]) < 58):
                                number_plate = upper_text + " " + lower_text
                                if number_plate in plateNumberDict.keys():
                                    plateNumberDict[str(number_plate)] += 1
                                else:
                                    plateNumberDict[str(number_plate)] = 1

            # else:
            #     number_plate = ""
            # print ("Number_Plate_Text = " + number_plate)
            # image = cv2.putText(image, number_plate, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        # cv2.imshow('crop', crop)
        # cv2.waitKey(0)
        if not not bool(plateNumberDict): 
            print("Dict: " + str(plateNumberDict))
            keyMax = max(plateNumberDict, key=plateNumberDict.get)
            print("KeyMax: " + str(keyMax))
            if plateNumberDict[keyMax] > 4:
                file = open("temp.txt", "w")
                file.write(keyMax + "-" + str(plateNumberDict[keyMax]))
                file.close()
                image = cv2.putText(image, keyMax + " XIN MOI QUET THE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

 # Important because if you don't close the file
    # The operating system will lock your file and your other scripts
    # Won't have access

# class MyFlaskApp(Flask):
#   def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
#     if not self.debug or os.getenv('WERKZEUG_RUN_MAIN') == 'true':
#       with self.app_context():
#         main()
#     super(MyFlaskApp, self).run(host='0.0.0.0', port=5000, debug=True, load_dotenv=load_dotenv, **options)

# app = MyFlaskApp(__name__)

# @app.route('/detections', methods=['GET'])
# def get_detections():
#     try:
#         return jsonify({"response": number_plate}), 200
#     except FileNotFoundError:
#         abort(404)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
