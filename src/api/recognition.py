from fastapi import APIRouter, Response, status
from fastapi import APIRouter, status, Response,UploadFile, File
from typing import List
from urllib.parse import unquote
from src.app.face_recognition import FaceRecognition
import cv2
import numpy as np
face_recognition = FaceRecognition()

router = APIRouter(prefix='/recognition', tags=['people'])

@router.post("",status_code=status.HTTP_200_OK)
async def recognize(image: UploadFile = File(...)):
	content = await image.read()
	image_buffer = np.frombuffer(content, np.uint8)
	np_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
	recognition_result = face_recognition.recognize(np_image)
	
	return recognition_result

@router.post("/training",status_code=status.HTTP_200_OK)
async def train():
	face_recognition.reload()


