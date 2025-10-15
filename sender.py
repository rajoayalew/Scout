import socket
import time
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

picam2 = Picamera2()
video_config = picam2.create_video_configuration({"size": (1280, 720)})
picam2.configure(video_config)
encoder = H264Encoder(1000000)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 10001))
    sock.listen()
    print("Waiting for client...")
    
    picam2.encoders = encoder
    conn, addr = sock.accept()
    print(f"Client connected: {addr}")
    stream = conn.makefile("wb")
    encoder.output = FileOutput(stream)

    try:
        picam2.start_encoder(encoder)
        picam2.start()
        print("Streaming video. Press Ctrl+C to stop.")
        
        while True:
            time.sleep(1)  # keep looping, camera keeps streaming

    except KeyboardInterrupt:
        print("Stopping stream...")

    finally:
        picam2.stop()
        picam2.stop_encoder()
        conn.close()
        print("Stream stopped, connection closed.")
