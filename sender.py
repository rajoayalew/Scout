import socket
import threading
import time
import json
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from gtts import gTTS
import os

def speak(txt, lang='en'):
    tts = gTTS(txt, lang='en')   
    tts.save("message.mp3")
    os.system(f"pw-play message.mp3")

def command_server():
    """Command listener on port 10002."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cmd_sock:
        cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        cmd_sock.bind(("0.0.0.0", 10002))
        cmd_sock.listen()
        print("Waiting for command client...")
        conn, addr = cmd_sock.accept()
        print(f"Command client connected: {addr}")

        # Turn the socket into a line-reader
        conn_file = conn.makefile("r")

        try:
            for raw in conn_file:       # reads one JSON object per line
                raw = raw.strip()
                if not raw:
                    continue

                print("Command received:", raw)
                data = json.loads(raw)  # <-- ALWAYS valid JSON now

                message = "SaysID "

                for item in data["objects"]:
                    print(item)

                    if item["mean_depth"] < 100:
                        continue

                    pos = item["position"]
                    name = item["name"]

                    if pos == "ahead":
                        message += f"{name} is ahead of you "
                    elif pos == "left":
                        message += f"{name} to the left of you "
                    elif pos == "right":
                        message += f"{name} to the right of you "
                
                if message == "SaysID ":
                    continue

                print(message)
                speak(message)

        except Exception as e:
            print("Command socket error:", e)

def main():
    # Start command socket thread
    threading.Thread(target=command_server, daemon=True).start()

    # Start video server
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration({"size": (1280, 720)})
    picam2.configure(video_config)
    encoder = H264Encoder(1000000)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as video_sock:
        video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        video_sock.bind(("0.0.0.0", 10001))
        video_sock.listen()
        print("Waiting for video client...")

        conn, addr = video_sock.accept()
        print(f"Video client connected: {addr}")
        stream = conn.makefile("wb")
        encoder.output = FileOutput(stream)

        try:
            picam2.start_encoder(encoder)
            picam2.start()
            print("Streaming video. Press Ctrl+C to stop.")

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("Stopping stream...")
        finally:
            picam2.stop()
            picam2.stop_encoder()
            conn.close()
            print("Video stream stopped.")


if __name__ == "__main__":
    main()
