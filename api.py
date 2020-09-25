# from flask import Flask, request, Response, jsonify, send_from_directory, abort
# import time
from absl import app, logging
import cv2
import numpy as np
import os
import requests

# # Initialize Flask application
# app = Flask(__name__)

# number = ""

# # API that returns JSON with classes found in images
# @app.route('/detections', methods=['GET'])
# def get_detections():
#     while True:
#         file = open("temp.txt", "r")
#         number = str(file.read())
#         try:
#             return jsonify({"response": number}), 200
#         except FileNotFoundError:
#             abort(404)

# if __name__ == '__main__':
#     app.run(debug=True, host = '0.0.0.0', port=5000)
def main(argv):    
    url = 'https://votan-sparking.herokuapp.com/tickets/createticket'
    userId = '17521022'
    while True:
        file = open("temp.txt", "r")
        number = str(file.read())
        if len(number) > 10:
            dataArr = number.split("-")
            if (int(dataArr[1]) == 5):
                payload = { 'numplate': dataArr[0], 'userId': userId}
                r = requests.post(url, data=payload)
                print('RESULT: ' + r.text)
                
if __name__ == '__main__':
    app.run(main)