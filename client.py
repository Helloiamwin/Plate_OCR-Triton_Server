import cv2
import numpy as np
import base64
import requests

url = 'http://192.168.0.118:8000/yolov5_onnx_inference'
file = {'file': open('images/zidane.jpg', 'rb')}
resp = requests.post(url=url, files=file) 
print(resp.json())
