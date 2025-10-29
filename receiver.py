import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

# List of models that can be used are given below
"""
['DPTDepthModel', 'DPT_BEiT_B_384', 'DPT_BEiT_L_384', 'DPT_BEiT_L_512', 
'DPT_Hybrid', 'DPT_Large', 'DPT_LeViT_224', 'DPT_Next_ViT_L_384', 'DPT_SwinV2_B_384', 
'DPT_SwinV2_L_384', 'DPT_SwinV2_T_256', 'DPT_Swin_L_384', 'MiDaS', 'MiDaS_small', 
'MidasNet', 'MidasNet_small', 'transforms']
"""

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda")
midas.to(device)
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

frameCount = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Original", frame)

    frameCount += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frameCount % 4 != 0:
        continue
    
    input_batch = transform(frame).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    depthNorm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depthColor = cv2.applyColorMap(depthNorm, cv2.COLORMAP_INFERNO)

    cv2.imshow("Depth", depthColor)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

