import json

from flask import Flask, request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util


tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


label_path = './mscoco_label_map.pbtxt'

path_saved_model = "./saved_model"

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(path_saved_model)
category_index = label_map_util.create_category_index_from_labelmap(label_path,use_display_name=True)

import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


#----------------read image and test--------------------#


app = Flask(__name__)


@app.route('/')
def index():
    return '<h1>Hello World!</h1>'

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    f = request.files.get('file')
    print(f)
    basepath = os.path.dirname(__file__)  # 当前文件所在路径
    upload_path = os.path.join("tmp/tmp." + f.filename.split(".")[-1])
                               #secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    return upload_path

@app.route('/detect')
def inference():
    im_url = request.args.get('url')
    image_np = cv2.imread(im_url)
    sp = image_np.shape
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    result_list = list()

    # print(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'])
    for i in range(len(detections['detection_scores'])):
        if detections['detection_scores'][i] > 0.4:
            bbox = detections['detection_boxes'][i]
            cate = detections['detection_classes'][i]
            y1 = bbox[0] * sp[0]
            x1 = bbox[1] * sp[1]
            y2 = bbox[2] * sp[0]
            x2 = bbox[3] * sp[1]
            result_dict = {}
            result_dict["left_up_x"] = x1
            result_dict["left_up_y"] = y1
            result_dict["right_down_x"] = x2
            result_dict["right_down_y"] = y2
            #print(cate)
            result_dict["class"] = str(cate)
            result_list.append(result_dict)

    final_result = {}
    final_result["object"] = result_list
    return json.dumps(final_result)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=80, debug=True)