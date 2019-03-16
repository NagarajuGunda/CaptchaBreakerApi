import logging
import random
import time
try:
    import Image
except ImportError:
    from PIL import Image
import io as IO
import base64
from PIL import ImageEnhance, ImageFilter
import cv2
import matplotlib.pyplot as plt
import glob
import random
import base64
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask import flash, redirect, render_template, request, session, abort
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
import cv2
import numpy as np
import os
import sys
import json

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

# run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict

app = Flask(__name__)
app.config.from_object(__name__)

import os
import glob

PORT_TO_RUN = 50009

def process_image(path):    
    img = Image.open(IO.BytesIO(base64.b64decode(str(path.split(",")[1])))).convert('RGBA')
    
    # new_size = (img.size[0] * 2, img.size[1] * 2)

    # Remove the grey portion
    newimdata = []
    datas = img.getdata()
    
    for item in datas:
        if item[0] < 112 or item[1] < 112 or item[2] < 112:
            newimdata.append(item)
        else:
            newimdata.append((255, 255, 255))
    img.putdata(newimdata)

    return img.convert('RGB')
    #return img.resize(new_size, Image.ANTIALIAS).convert('RGB')

# Model preparation 
FASTER_RCNN_INCEPTION_V2_FROZEN_MODEL = 'frozen_inference_graphs/faster_rcnn_inception_v2/frozen_inference_graph.pb'
SSD_INCEPTION_V2_FRONZEN_MODEL = 'frozen_inference_graphs/ssd_inception_v2/frozen_inference_graph.pb'
PATH_TO_FROZEN_GRAPH = SSD_INCEPTION_V2_FRONZEN_MODEL
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'labelmap.pbtxt'
NUM_CLASSES = 37

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    sess = tf.Session(graph=detection_graph)

def Captcha_detection(image, average_distance_error=3, probability=0.65):
    start_time = time.time()
    # Resize image if needed
    #image_np = cv2.resize(np.array(image), (0,0), fx=3, fy=3) 
    # To get real color we do this:
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    # print("Before Session Run --- %s seconds ---" % (time.time() - start_time))
    
    # Visualization of the results of a detection.
    (boxes, scores, classes, num_detections) = sess.run(
      [boxes, scores, classes, num_detections],
      feed_dict={image_tensor: image_np_expanded})
    
    # print("After Session Run --- %s seconds ---" % (time.time() - start_time))

    # Bellow we do filtering stuff
    captcha_array = []
    # loop our all detection boxes
    for i,b in enumerate(boxes[0]):
        for Symbol in range(NUM_CLASSES):
            if classes[0][i] == Symbol: # check if detected class equal to our symbols
                if scores[0][i] >= probability: # do something only if detected score more han 0.65
                                    # x-left        # x-right
                    mid_x = (boxes[0][i][1]+boxes[0][i][3])/2 # find x coordinates center of letter
                    # to captcha_array array save detected Symbol, middle X coordinates and detection percentage
                    captcha_array.append([category_index[Symbol].get('name'), mid_x, scores[0][i]])

    # rearange array acording to X coordinates datected
    for number in range(20):
        for captcha_number in range(len(captcha_array)-1):
            if captcha_array[captcha_number][1] > captcha_array[captcha_number+1][1]:
                temporary_captcha = captcha_array[captcha_number]
                captcha_array[captcha_number] = captcha_array[captcha_number+1]
                captcha_array[captcha_number+1] = temporary_captcha


    # Find average distance between detected symbols
    average = 0
    captcha_len = len(captcha_array)-1
    while captcha_len > 0:
        average += captcha_array[captcha_len][1]- captcha_array[captcha_len-1][1]
        captcha_len -= 1
    # Increase average distance error
    average = average/(len(captcha_array)+average_distance_error)

    
    captcha_array_filtered = list(captcha_array)
    captcha_len = len(captcha_array)-1
    while captcha_len > 0:
        # if average distance is larger than error distance
        if captcha_array[captcha_len][1]- captcha_array[captcha_len-1][1] < average:
            # check which symbol has higher detection percentage
            if captcha_array[captcha_len][2] > captcha_array[captcha_len-1][2]:
                del captcha_array_filtered[captcha_len-1]
            else:
                del captcha_array_filtered[captcha_len]
        captcha_len -= 1

    # Get final string from filtered CAPTCHA array
    captcha_string = ""
    for captcha_letter in range(len(captcha_array_filtered)):
        captcha_string += captcha_array_filtered[captcha_letter][0]
        
    # print("Actual Filtering --- %s seconds ---" % (time.time() - start_time))
    print("Execution time: %s seconds" % (time.time() - start_time))
    app.logger.info("Execution time: %s seconds" % (time.time() - start_time))
    app.logger.info("Image classified as %s" % (captcha_string))
    return captcha_string

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        captcha = request.form['captcha']
        image = process_image(captcha)
        result = Captcha_detection(image, 6, 0.7)
    return render_template('index.html', **locals())

@app.route('/predict/', methods=["POST"])
def classify():
    if not request.json:
        abort(400)
    image_path = request.json['captcha']
    app.logger.info("Classifying image %s" % (image_path),)

    # Load in an image to classify and preprocess it
    image = process_image(image_path)
    
    # Get the predictions (output of the softmax) for this image
    t = time.time()
    preds = Captcha_detection(image, 6, 0.7)
    dt = time.time() - t
    app.logger.info("Execution time: %0.2f" % (dt * 1000.))
    
    app.logger.info("Image %s classified as %s" % (image_path, preds))

    return preds

if __name__ == '__main__':
    port = int(os.environ.get('PORT', PORT_TO_RUN))
    app.run(host='0.0.0.0', port=port)
