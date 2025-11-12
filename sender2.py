import socket
import time
from threading import Thread
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import pyttsx3

VIDEO_PORT = 10001
NAME_PORT = 10002
HOST = "0.0.0.0"

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # speaking speed
tts_engine.setProperty('volume', 1.0)  # max volume

# Uses pyttsx3 to do text-to-speech which plays out through 
# default audio device defined in the OS
def speak_name(name):
    tts_engine.say(name)
    tts_engine.runAndWait()

def receive_names(conn):
    with conn.makefile("rb") as f:  # buffered reader for line-based reading
        while True:
            try:
                line = f.readline()
                if not line:
                    print("[Names] Connection closed by client")
                    break
                name = line.decode().strip()
                if name:
                    print(f"[Names] Received object name: {name}")
                speak_name(name)
            except Exception as e:
                print(f"[Names] Error receiving name: {e}")
                break

def video_stream():
    # Camera setup
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration({"size": (1280, 720)})
    picam2.configure(video_config)
    encoder = H264Encoder(1000000)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, VIDEO_PORT))
        sock.listen(1)
        print(f"[Video] Waiting for client on port {VIDEO_PORT}...")

        conn, addr = sock.accept()
        print(f"[Video] Client connected: {addr}")

        stream = conn.makefile("wb")
        encoder.output = FileOutput(stream)
        picam2.encoders = encoder

        try:
            picam2.start_encoder(encoder)
            picam2.start()
            print("[Video] Streaming video. Press Ctrl+C to stop.")

            while True:
                time.sleep(1)  # keep looping, camera keeps streaming

        except KeyboardInterrupt:
            print("[Video] Stopping video stream...")

        finally:
            picam2.stop()
            picam2.stop_encoder()
            conn.close()
            print("[Video] Stream stopped, connection closed.")

def names_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, NAME_PORT))
        sock.listen(1)
        print(f"[Names] Waiting for client on port {NAME_PORT}...")

        conn, addr = sock.accept()
        print(f"[Names] Client connected: {addr}")

        # Start receiving names
        receive_names(conn)


t1 = Thread(target=video_stream, daemon=True)
t2 = Thread(target=names_server, daemon=True)

t1.start()
t2.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Server stopped.")
