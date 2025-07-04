import numpy as np
import sys
import cv2
import torch
import copy
import tritonclient.grpc as grpcclient
url = "192.168.0.118:8001"
verbose = False
model_name = "python_backend_for_test"

try:
    triton_client = grpcclient.InferenceServerClient(url=url,verbose=verbose)
except Exception as e:
    print("channel creation failed: " + str(e))
    sys.exit()
imgs = []
h,w,bz = 640, 640, 1

img1 = cv2.imread("images/zidane.jpg")
img1 = cv2.resize(img1, (w, h))

img2 = cv2.imread("images/Conte.jpg")
img2 = cv2.resize(img2, (w, h))

imgs.append(img1)
imgs.append(img2)

bz = len(imgs)

imgs = np.array(imgs)
imgs = np.transpose(imgs, (0, 3,1,2))

inputs = []
outputs = []
inputs.append(grpcclient.InferInput('images', [bz,3,h,w], "UINT8"))
outputs.append(grpcclient.InferRequestedOutput('output'))

inputs[0].set_data_from_numpy(imgs)

results = triton_client.infer(model_name=model_name,
                                inputs=inputs,
                                outputs=outputs)

imgs = results.as_numpy('output')
for i,img in enumerate(imgs):
    cv2.imwrite(f"result_{i}.jpg", img)

