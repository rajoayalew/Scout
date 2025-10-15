import cv2

# GStreamer pipeline for receiving raw H.264 over TCP
pipeline = (
    'tcpclientsrc host=192.168.1.85 port=10001 ! '
    'h264parse ! avdec_h264 ! videoconvert ! appsink'
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()