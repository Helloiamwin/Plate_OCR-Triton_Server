from typing import Union
from fastapi import FastAPI, File, UploadFile
import onnxruntime as rt
import numpy as np 
import base64
import torch
import numpy
import cv2
import yaml
import faiss
import json
from collections import Counter
from src.app.face_recognition import FaceRecognition
# from src.models.yolov5.yolov5_onnx import YOLOV5Onnx
# from src.models.yolov5.yolov5_utils import  (letterbox,
#                                             non_max_suppression,
#                                             scale_boxes)


# config_fp = "configs/yolov5_onnx.yaml"
# with open(config_fp, 'r') as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
# yolov5_onnx = YOLOV5Onnx(config)




# @app.post("/yolov5_onnx_inference")
# async def yolov5_onnx_inference(file: UploadFile):
#     content = await file.read()
#     image_buffer = np.frombuffer(content, np.uint8)
#     image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

#     preds = yolov5_onnx.inference(image)

#     results = {}
#     for i, det in enumerate(preds):
#         objects = []
#         for box in det:
#             obj = {}
#             obj["bbox"] = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
#             obj["score"] = str(box[4])
#             obj["class"] = int(box[5])
#             objects.append(obj)

#         results[str(i)] = objects

#     return results

face_rec = FaceRecognition("configs/face_recognition.yaml")
index = faiss.read_index("face_embedding.index")

app = FastAPI()
@app.post("/face_recognition")
async def face_recognition(file: UploadFile):
    content = await file.read()
    image_buffer = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

    with open('database.json') as user_file:
        data = json.load(user_file)

    encodes = []
    person_ids = []
    for key, vectors in data.items():
        for vector in vectors:
            encodes.append(np.array(vector, dtype=np.float32))
            person_ids.append(key)

    _, _, encodes =  face_rec.get_encode(image)
    D, I  = index.search(encodes, 5)
    person_list = []
    for i, dis in enumerate(D[0]):
        if dis > 0.5:
            person_list.append(person_ids[I[0][i]])
    if len(person_list):
        counter = Counter(person_list)
        most_common = counter.most_common(1)[0][0]
        result = {"person_id": most_common}
    else:
        result = {"person_id": "unknow"}
    return result



