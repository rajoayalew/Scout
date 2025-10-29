import cv2
import torch
import numpy as np
from threading import Thread, Lock
from ultralytics import YOLO

# List of models that can be used are given below
"""
['DPTDepthModel', 'DPT_BEiT_B_384', 'DPT_BEiT_L_384', 'DPT_BEiT_L_512', 
'DPT_Hybrid', 'DPT_Large', 'DPT_LeViT_224', 'DPT_Next_ViT_L_384', 'DPT_SwinV2_B_384', 
'DPT_SwinV2_L_384', 'DPT_SwinV2_T_256', 'DPT_Swin_L_384', 'MiDaS', 'MiDaS_small', 
'MidasNet', 'MidasNet_small', 'transforms']
"""

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

# GStreamer pipeline for receiving raw H.264 over TCP
pipeline = (
    'tcpclientsrc host=192.168.1.85 port=10001 ! '
    'h264parse ! avdec_h264 ! videoconvert ! appsink'
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open stream")
    exit()

# Shared variables
latestFrame = None      # latest camera frame
latestDepth = None      # latest depth map
latestYOLO  = None     # latest detection frame
frameLock = Lock()      # to prevent race conditions

def capture_thread():
    global latestFrame

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        with frameLock:
            latestFrame = frame.copy()


def depth_thread():
    global latestFrame, latestDepth

    frameCount = 0

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

            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            latestDepth = depth_color

def detection_thread():
    global latestFrame, latestYOLO, threshold

    while True:
        if latestFrame is not None:
            with frameLock:
                frame_copy = latestFrame.copy()

            results = yolo_model.predict(
                source=frame_copy,
                device=device,
                imgsz=960,      # larger input for better detection
                conf=0.25,      # lower confidence threshold
                half=True       # FP16 for speed
            )

            result_frame = frame_copy.copy()
            for result in results:
                for box in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = box
                    if score > threshold:
                        cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                        cv2.putText(result_frame, result.names[int(class_id)].upper(),
                                    (int(x1), int(y1-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            with frameLock:
                latestYOLO = result_frame.copy()

# Start threading
t1 = Thread(target=capture_thread, daemon=True)
t2 = Thread(target=depth_thread, daemon=True)
t3 = Thread(target=detection_thread, daemon=True)

t1.start()
t2.start()
t3.start()

while True:
    displayFrame = None
    displayDepth = None
    displayYOLO = None

    with frameLock:
        if latestFrame is not None:
            displayFrame = latestFrame.copy()
        if latestDepth is not None:
            displayDepth = latestDepth.copy()
        if latestYOLO is not None:
            displayYOLO = latestYOLO.copy()

    if displayFrame is not None:
        cv2.imshow("Original", displayFrame)
    if displayDepth is not None:
        cv2.imshow("Depth", displayDepth)
    if displayYOLO is not None:
        cv2.imshow("YOLO View", displayYOLO)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
