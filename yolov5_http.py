import argparse
import numpy as np
import sys
import cv2
from src.model.yolov5.yolov5_utils import letterbox, non_max_suppression
import torch
import copy

import tritonclient.http as httpclient
url = "192.168.0.118:8000"
verbose = False
model_name = "yolov5_onnx"

try:
    triton_client = httpclient.InferenceServerClient(url=url,verbose=verbose)
except Exception as e:
    print("channel creation failed: " + str(e))
    sys.exit()

# Health
if not triton_client.is_server_live():
    print("FAILED : is_server_live")
    sys.exit(1)

if not triton_client.is_server_ready():
    print("FAILED : is_server_ready")
    sys.exit(1)

if not triton_client.is_model_ready(model_name):
    print("FAILED : is_model_ready")
    sys.exit(1)

metadata = triton_client.get_server_metadata()
if not (metadata['name'] == 'triton'):
    print("FAILED : get_server_metadata")
    sys.exit(1)

# Configuration
config = triton_client.get_model_config(model_name)
if not (config["name"] == model_name):
    print("FAILED: get_model_config")
    sys.exit(1)

inputs = []
outputs = []
inputs.append(httpclient.InferInput('images', [1,3,640,640], "FP32"))
outputs.append(httpclient.InferRequestedOutput('output0'))

img = cv2.imread("images/zidane.jpg")
img = letterbox(img, (640,640), stride=32, auto=False)[0]
img = img.transpose((2, 0, 1))[::-1]
img = img / 255
imgs = np.expand_dims(img, axis=0).astype(np.float32)

inputs[0].set_data_from_numpy(imgs)

results = triton_client.infer(model_name=model_name,
                                inputs=inputs,
                                outputs=outputs)

ori_output0 = results.as_numpy('output0')
output0 = copy.deepcopy(ori_output0)
output0 = [torch.from_numpy(output0)]
output0 = non_max_suppression(output0, 0.25, 0.45, None, False, 1000)
print(output0)

