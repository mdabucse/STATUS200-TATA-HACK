import numpy as np
import cv2
from PIL import Image
from time import time

import tflite_runtime.interpreter as tflite

def processImg(img):
    img_tensor = np.array(img).astype(np.float32)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return img_tensor

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

interpreter = tflite.Interpreter(model_path="blind_final.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cams = [cv2.VideoCapture(0), cv2.VideoCapture(2)]

for i, cam in enumerate(cams):
    if not cam.isOpened():
        print(f"Couldn't open camera {i}")
        exit()

font = cv2.FONT_HERSHEY_SIMPLEX

def getPredictionFromRetAndFrame(ret, frame, winName):
    if not ret:
        print("Can't receive frame, exiting...")
        exit()
    
    timer = time()

    LINE_THICKNESS = 10
    LINE_COLOUR    = (0, 0, 255)
    ALPHA          = 0.5
    
    cv2.line(frame,   (0,     678),     (1280, 450),    LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1280,  450),     (1750, 450),    LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1280,  450),     (1280, 656),    LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1750,  450),     (1739, 627),    LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1739,  627),     (1254, 1080),   LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1280,  656),     (0,    1080),   LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1280,  656),     (1739,  627),   LINE_COLOUR,    thickness=LINE_THICKNESS)
    
    cropped = frame[360:1080, 0:1920]
    resized = cv2.resize(cropped, (160, 160))

    interpreter.set_tensor(input_details[0]['index'], processImg(resized))
    interpreter.invoke()

    pred_raw = interpreter.get_tensor(output_details[0]['index'])
    pred_sig = sigmoid(pred_raw)
    pred = np.where(pred_sig < 0.5, 0, 1)
    timer = time() - timer

    readable_val = winName if pred[0][0] == 0 else ""
    print(readable_val)
    print("----------------------\n\n")

cameraNames = ["Left", "Right"]

while True:
    for i in range(len(cams)):
        ret, frame = cams[i].read()
        getPredictionFromRetAndFrame(ret, frame, cameraNames[i])

    if cv2.waitKey(1) == ord('q'):
        break
