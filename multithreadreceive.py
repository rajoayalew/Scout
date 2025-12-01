import cv2
import torch
import numpy as np
from threading import Thread, Lock
from ultralytics import YOLO
from collections import deque
import copy
import socket
import json
import time

## Setup of Models

threshold = 0.5 # Defines what threshold score is required for an item to be recognized
model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

yolo_model = YOLO("yolo11l.pt")

device = torch.device("cuda")
midas.to(device)
yolo_model.to(device)

midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

## Setup of Queue

recentNames = deque(maxlen=10)

## Setup of Video TCP Server

# GStreamer pipeline for receiving raw H.264 over TCP
pipeline = (
    'tcpclientsrc host=192.168.1.162 port=10001 ! '
    'h264parse ! avdec_h264 ! videoconvert ! appsink'
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open stream")
    exit()

# Shared variables between threads

latestFrame = None      # latest camera frame
latestDepth = None      # latest depth map
latestYOLO  = None     # latest detection frame
latestResult = []
frameCount = 0

frameLock = Lock()

## Setup of Object Name TCP Server

name_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
name_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
name_sock.connect(("192.168.1.162", 10002))


# Sends the name of the closest object detected to the Raspberry Pi via the name_sock
# TCP socket
def send_closest_item(depthImg, yoloResult):

    resultsList = []

    if (len(yoloResult) == 0):
        return

    for result in yoloResult:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box

            if score > threshold:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                name = result.names[class_id].upper()

                depthBox = depthImg[y1:y2, x1:x2]
                avgX = (x1 + x2) / 2
                meanDepthScore = np.mean(depthBox)

                position = None
                if (0 <= avgX and avgX <= 426):
                    position = "left"
                elif (426 <= avgX and avgX <= 853):
                    position = "ahead"
                elif (853 <= avgX and avgX <= 1280):
                    position = "right"

                resultsList.append({"name": name,
                                    "mean_depth": meanDepthScore,
                                    "position": position,
                                    "avgX": avgX})  
    if (not resultsList):
        return        
    
    print(resultsList)                
    message = json.dumps({"objects": resultsList})
    name_sock.sendall((message+"\n").encode("utf-8"))

def capture_thread():
    global latestFrame

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        with frameLock:
            latestFrame = frame.copy()

# Thread used to run MiDaS analysis to determine distance of object from camera
def depth_thread():
    global latestFrame, latestDepth, frameCount

    while True:

        if latestFrame is not None:
            frameCount += 1

            if frameCount % 4 != 0:
                continue  # only process every 4th frame

            with frameLock:
                frame_rgb = cv2.cvtColor(latestFrame, cv2.COLOR_BGR2RGB)

            input_batch = transform(frame_rgb).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth = prediction.cpu().numpy()
            depth_color = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            with frameLock:
                latestDepth = depth_color.copy()

# Thread used to run 
def detection_thread():
    global latestFrame, latestYOLO, latestResult, threshold

    while True:

        if latestFrame is not None:
            with frameLock:
                frame_copy = latestFrame.copy()

            results = yolo_model.predict(
                source=frame_copy,
                device=device,
                imgsz=960,      # larger input for better detection
                conf=0.25,      # lower confidence threshold
                half=True,       # FP16 for speed
                verbose=False
                #tracker="bytetrack.yaml"
            )

            for result in results:
                for box in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = box
                    if score > threshold:
                        x1, y1, x2, y2, class_id = map(int, [x1, y1, x2, y2, class_id])
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0,255,0), 2)
                        name = result.names[class_id].upper()
                        cv2.putText(frame_copy, "{} {}".format(name, score),
                                    (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                
            with frameLock:
                latestYOLO = frame_copy.copy()
                latestResult = copy.deepcopy(result)

# Start threading
t1 = Thread(target=capture_thread, daemon=True)
t2 = Thread(target=depth_thread, daemon=True)
t3 = Thread(target=detection_thread, daemon=True)

t1.start()
t2.start()
t3.start()

startTime = time.perf_counter()

while True:
    displayFrame = None
    displayDepth = None
    displayYOLO = None
    currResult = []
    localFrameCount = None

    with frameLock:
        if latestFrame is not None:
            displayFrame = latestFrame.copy()
        if latestDepth is not None:
            displayDepth = latestDepth.copy()
        if latestYOLO is not None:
            displayYOLO = latestYOLO.copy()

        if (len(latestResult) != 0):
            currResult = copy.deepcopy(latestResult)

        localFrameCount = frameCount

    if displayFrame is not None:
        cv2.imshow("Original", displayFrame)
    if displayDepth is not None:
        cv2.imshow("Depth", displayDepth)
    if displayYOLO is not None:
        cv2.imshow("YOLO View", displayYOLO)

    currTime = time.perf_counter()
    elapsed = currTime - startTime
    print(elapsed)

    if (localFrameCount % 4 == 0 and displayDepth is not None and currResult is not None and elapsed > 5):

        if (len(currResult) != 0):
            startTime = time.perf_counter()
            send_closest_item(displayDepth, currResult)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
