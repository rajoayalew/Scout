import socket
import time
from threading import Thread
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

def receive_names(conn):
    """
    Thread function to continuously receive object names from the client.
    """
    with conn.makefile("rb") as f:  # buffered reader for line-based reading
        while True:
            try:
                line = f.readline()
                if not line:
                    break  # connection closed
                name = line.decode().strip()
                if name:
                    print(f"Received object name: {name}")
            except Exception as e:
                print(f"Error receiving name: {e}")
                break

# Camera setup
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
    
    # Start thread for receiving object names
    recv_thread = Thread(target=receive_names, args=(conn,), daemon=True)
    recv_thread.start()
    
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
