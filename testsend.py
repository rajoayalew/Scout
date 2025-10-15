#!/usr/bin/python3

import socket
import time

UDP_IP = "192.168.1.67"  # Replace with your laptop's IP
UDP_PORT = 10001

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

for i in range(20):
    message = f"Hello {i}"
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
    print(f"Sent: {message}")
    time.sleep(0.5)
