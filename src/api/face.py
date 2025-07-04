from urllib.parse import unquote
from fastapi import APIRouter, status, Response,UploadFile, File
from fastapi.responses import JSONResponse
from src.schemas.validation import ImageValidation
from src.app.face_crud import FaceCRUD
import cv2
import numpy as np
# from models import FaceDoc
from typing import List

face_crud_instance = FaceCRUD()

router = APIRouter(prefix="/people", tags=['faces'])

@router.get(
	"/{person_id}/faces",
	status_code=status.HTTP_200_OK
)
async def get_all_faces(person_id: str, skip: int = 0, limit: int = 10):
	face_docs = face_crud_instance.select_all_face_of_person(person_id, skip, limit)
	return face_docs

@router.post(
	"/{person_id}/faces",
	response_model=ImageValidation,
	status_code=status.HTTP_201_CREATED
)
async def insert_one_face(person_id: str, image: UploadFile = File(...)):
	content = await image.read()
	image_buffer = np.frombuffer(content, np.uint8)
	np_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
	person_id = unquote(person_id)
	res = face_crud_instance.insert_face(person_id, np_image)
	return res


@router.delete(
	"/{person_id}/faces/{face_id}",
	response_class=Response,
	status_code=status.HTTP_204_NO_CONTENT
)
async def delete_one_face(person_id: str, face_id: str):
	face_crud_instance.delete_face_by_id(person_id, face_id)

@router.delete(
	"/{person_id}/faces",
	response_class=Response,
	status_code=status.HTTP_204_NO_CONTENT
)
def delete_all_faces(person_id: str):
	face_crud_instance.delete_all_face(person_id)