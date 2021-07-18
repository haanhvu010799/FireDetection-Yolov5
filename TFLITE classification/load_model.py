import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import os
import cv2
import sys

from PIL import Image
import numpy as np
from numpy import array
from imutils import paths
import pathlib

if __name__ == "__main__":

    rows = 224
    cols = 224

    interpreter = Interpreter(model_path="./quant_pruned_firenet.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(sys.argv) == 1:

        video = cv2.VideoCapture(0)
        print("Loaded camera")

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:

            ret, frame1 = video.read()
            if not ret:
                print("EOF")
                break;
            
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            input_frame = np.expand_dims(frame_resized, axis=0)
            input_frame = np.array(input_frame, dtype=np.float32)

            interpreter.set_tensor(input_details[0]['index'], input_frame)
            interpreter.invoke()            
            output = interpreter.get_tensor(output_details[0]['index']) 
            
            if round(output[0][0]) == 1:
                print("Đang cháy")
            else:
                print("Không cháy")
