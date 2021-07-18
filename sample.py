# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.
TEST_DATA=../all_models
Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt
"""
import argparse
import collections
import common
import cv2
import numpy as np
import os
from PIL import Image
import re
import tflite_runtime.interpreter as tflite
import time
# from sendDataToThingsboard import sendDataToThingsboard
# connect to thingsboard
import sys
import paho.mqtt.client as mqtt
import json

# streaming packages
from flask import Flask, render_template, Response
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')


Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))
    # print('count',count)
    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))
    vehicles = []
    for i in range(count):
        if (scores[i] >= score_threshold):
             vehicles.append(make(i))
    return vehicles


vehiclesPassLine1 = {}
vehiclesPassLine2 = {}
vehiclesInMoiVideo0={}

totalBusPassLine1 = 0
totalCarPassLine1 = 0
totalMotorcyclePassLine1 = 0
totalTruckPassLine1 = 0

totalBusPassLine2 = 0
totalCarPassLine2 = 0
totalMotorcyclePassLine2 = 0
totalTruckPassLine2 = 0

busInMoiVideo0=0
carInMoiVideo0=0
motorcycleInMoiVideo0=0
truckInMoiVideo0=0

def append_objs_to_img(cv2_im, objs, labels,client):
    height, width, channels = cv2_im.shape
    global totalBusPassLine1 
    global totalCarPassLine1 
    global totalMotorcyclePassLine1 
    global totalTruckPassLine1

    global totalBusPassLine2 
    global totalCarPassLine2 
    global totalMotorcyclePassLine2
    global totalTruckPassLine2 

    global busInMoiVideo0
    global carInMoiVideo0
    global motorcycleInMoiVideo0
    global truckInMoiVideo0

   
    global vehiclesPassLine2
    global vehiclesPassLine1
    global vehiclesInMoiVideo0

    busInMoiVideo0 =0
    carInMoiVideo0 =0
    motorcycleInMoiVideo0=0
    truckInMoiVideo0=0
    
    passLine1 = False
    passLine2 = False
    for obj in objs:
        color = (0, 0, 255)  
        isClosed= True
        
        # ve khung cho  Video0
        pts= np.array([[8,441],[439,150],[812,150],[1225,441]],np.int32)
        pts= pts.reshape((-1,1,2))
        cv2.polylines(cv2_im, [pts] ,isClosed ,color, 2)
        cv2.line(cv2_im, (632,151),(632,444), color,2)

        # label cho bounding box   
        label = 'id : {}'.format(str(obj.id))

        # ve tam cho bounding box cua doi tuong
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        cen_x= round((x0+x1)/2.0)
        cen_y= round((y0+y1)/2.0)
        center = (cen_x, cen_y)
        
        #Loại 0: xe trên 7 chỗ như xe buýt, xe khách
        #Loại 1: xe 4-7 chỗ như xe hơi, taxi, xe bán tải…
        #Loại 2: xe 2 bánh như xe đạp, xe máy
        #Loại 3: xe tải, container, xe cứu hỏa
        if((260<cen_y<=441 and cen_x<=1226) or (150<cen_y<=260 and cen_x<=967)):
            if(obj.id==0):
                color = (14,111,234) # xe bus cam
                busInMoiVideo0+=1
            elif(obj.id==1):
                color =  (224,100,2) # xe oto xanh duong 
                carInMoiVideo0+=1
            elif(obj.id==2):
                color =  (28,225,62) # xe may xanh la cay
                motorcycleInMoiVideo0+=1
            else:
                color = (240,2,165) # xe tai tim
                truckInMoiVideo0+=1
        
            vehiclesInMoiVideo0 = {
                "busInMoiVideo0": busInMoiVideo0,
                "carInMoiVideo0": carInMoiVideo0,
                "motorcycleInMoiVideo0": motorcycleInMoiVideo0,
                "truckInMoiVideo0": truckInMoiVideo0
            }   
            #  send so luong vehicles in MOI to thingsboard 
            print(vehiclesInMoiVideo0)
            client.publish('v1/devices/me/telemetry',json.dumps(vehiclesInMoiVideo0))

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), color, 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2_im = cv2.circle(cv2_im, center, 7, color, -1)
        # Dem xe neu vuot qua laser line 1
        elif(cen_x <= 633 and cen_y >441 and (cen_y - 441 ) <= 30):
            passLine1=True
            if(obj.id==0):
                totalBusPassLine1+=1
            elif(obj.id==1):
                totalCarPassLine1+=1
            elif(obj.id==2):
               totalMotorcyclePassLine1+=1
            else:
                totalTruckPassLine1+=1
            vehiclesPassLine1 = {
                "totalBusPassLine1": totalBusPassLine1,
                "totalCarPassLine1": totalCarPassLine1,
                "totalMotorcyclePassLine1": totalMotorcyclePassLine1,
                "totalTruckPassLine1": totalTruckPassLine1
            }     
            # send so luong vehicles pass line to thingsboard
            # print(vehiclesPassLine1)
            client.publish('v1/devices/me/telemetry',json.dumps(vehiclesPassLine1))
            

        # Dem xe neu vuot qua laser line 2
        elif(cen_x > 633 and cen_y < 150 and (150 - cen_y ) <= 3):
            passLine2=True
            if(obj.id==0):
                totalBusPassLine2+=1
            elif(obj.id==1):
                totalCarPassLine2+=1
            elif(obj.id==2):
               totalMotorcyclePassLine2+=1
            else:
                totalTruckPassLine2+=1
            vehiclesPassLine2 = {
                "totalBusPassLine2": totalBusPassLine2,
                "totalCarPassLine2": totalCarPassLine2,
                "totalMotorcyclePassLine2": totalMotorcyclePassLine2,
                "totalTruckPassLine2": totalTruckPassLine2
            }     
            # send so luong vehicles pass line to thingsboard
            # print(vehiclesPassLine2)
            client.publish('v1/devices/me/telemetry',json.dumps(vehiclesPassLine2))
    
    # Chop laser line
    if(passLine1):
        cv2.line(cv2_im, (8,441), (632,441),(0,255,0),3)
    if(passLine2):
        cv2.line(cv2_im, (634,150), (812,150),(0,255,0),3)
    return cv2_im


def main():

        THINGSBOARD_HOST = '127.0.0.1'
        ACCESS_TOKEN = 'UjMB6J3TJl9HGAmv1IMa'
 
        # connect to thingsboard
        client = mqtt.Client()

        client.username_pw_set(ACCESS_TOKEN)
        client.connect(THINGSBOARD_HOST, 1883)
        client.loop_start()

        # run model inference
        default_model_dir = '../all_models'
        default_model = 'ssd_mobiledet_int8.tflite'
        default_labels = 'vehicles_label.txt'
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
        parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
        parser.add_argument('--threshold', type=float, default=0.35,
                        help='classifier score threshold')
        args = parser.parse_args()

        print('Loading {} with {} labels.'.format(args.model, args.labels))
        interpreter = common.make_interpreter(args.model)
        interpreter.allocate_tensors()
        labels = load_labels(args.labels)

        cap = cv2.VideoCapture("Video0.mp4")
    
        while (cap.isOpened()):
            global frame_count
            ret, frame = cap.read()
            if ret==False:
                break
       

            # resize image
            cv2_im = frame
            cv2_im = cv2.resize(cv2_im, (1280,720))
        
            # 
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_rgb)
       
            # detect phuong tien
            common.set_input(interpreter, pil_im)
            interpreter.invoke()
            objs = get_output(interpreter, score_threshold=args.threshold)

            # append objs to img, count and send data to thingsboard
            cv2_im = append_objs_to_img(cv2_im, objs, labels,client)
        

            cv2.imwrite('demo.jpg', cv2_im)
            # cv2.imshow('frame',cv2_im)
            # streaming video to flask
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


        client.loop_stop()
        client.disconnect()



@app.route('/video_feed')
def video_feed():
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(host="192.168.100.181", port="8000", debug=True, threaded=True, use_reloader=True) 
    main()


# percent = int(100 * obj.score)
# label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
# label=labels.get(obj.id, obj.id)