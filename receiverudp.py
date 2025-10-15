import socket
import cv2
import numpy as np

UDP_IP = "0.0.0.0"   # listen on all interfaces
UDP_PORT = 10001

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)

print("Waiting for UDP video stream...")

# We'll accumulate chunks here
buffer = b""

try:
    while True:
        try:
            data, addr = sock.recvfrom(65536)  # max UDP packet size
            if not data:
                continue

            # Append data to buffer
            buffer += data

            # Decode the frame using OpenCV
            nparr = np.frombuffer(buffer, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("UDP Video", frame)
                buffer = b""  # reset buffer after successful decode

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except socket.timeout:
            continue

except KeyboardInterrupt:
    print("Stopping video...")

finally:
    sock.close()
    cv2.destroyAllWindows()
