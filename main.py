
import onnxruntime as rt
import numpy as np 
import torch
import numpy
import cv2
import yaml
import torch
import faiss
import os
import json
from src.model.scrfd.scrfd import SCRFD
from src.model.arcface.arcface import ArcFace
from src.model.utils.face_detect import Face
from src.utils.config import Config
from src.app.face_recognition import FaceRecognition
from src.app.person_crud import PersonCRUD
from src.model.scrfd.scrfd_trt import SCRFDTRT
from src.manager.detection_manager import DetectionManager

if __name__ == "__main__":

    img = cv2.imread("images/zidane03.jpg")
    face_recognition =  FaceRecognition()
    ret = face_recognition.recognize(img)
    print(ret)

    





