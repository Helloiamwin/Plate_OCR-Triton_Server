from src.database.person_db import PersonDatabase
from src.schemas.validation import ImageValidation
from src.app.face_recognition import FaceRecognition
from src.utils.config import Config
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
import os, shutil, cv2, uuid
from pathlib import Path
import numpy as np

class FaceCRUD:
	def __init__(self) -> None:
		self.db_instance = PersonDatabase()
		self.database_config = Config().get_database()
		self.face_recognition = FaceRecognition()
		
	def insert_face(self, person_id: str, image: np.ndarray):
		if not self.db_instance.check_person_by_id(person_id):
			raise HTTPException(status.HTTP_404_NOT_FOUND)
		collection = self.db_instance.get_person_collection()
		person_doc = collection.find_one({"id": person_id}, {"_id": 0})
		
		validate_result = ImageValidation.IMAGE_IS_VALID
		if validate_result == ImageValidation.IMAGE_IS_VALID:
			_, _, encodes = self.face_recognition.get_encode(image, True)
			vector_id = str(uuid.uuid4().hex)
			face_dir = os.path.join(self.database_config["path"], person_id)
			Path(face_dir).mkdir(parents=True, exist_ok=True)
			image_path = os.path.join(face_dir, f"{vector_id}{self.database_config['end']}")
			cv2.imwrite(image_path, image)
			embed_doc = {"id": vector_id,"img_path": image_path, "vector": encodes[0].tolist()}
			if "faces" not in person_doc.keys() or person_doc["faces"] is None:
				collection.update_one(
					{"id": person_id}, {"$set": {"faces": [embed_doc]}}
				)
			else:
				collection.update_one(
					{"id": person_id}, {"$push": {"faces": embed_doc}}
				)
			status_res = status.HTTP_201_CREATED
		else:
			status_res = status.HTTP_406_NOT_ACCEPTABLE
		return JSONResponse(
			status_code=status_res,
			content={"INFO": validate_result}
		)

	def select_all_face_of_person(self, person_id: str, skip: int, limit: int):
		if not self.db_instance.check_person_by_id(person_id):
			raise HTTPException(status.HTTP_404_NOT_FOUND)
		collection = self.db_instance.get_person_collection()
		person_doc = collection.find_one({"id": person_id}, {"faces.vector": 0})
		if "faces" not in person_doc.keys() or person_doc["faces"] is None:
			return []

		faces = person_doc["faces"]
		if skip < 0:
				skip = 0
		elif skip > len(faces):
				skip = len(faces) - 1
		if limit < 0:
				limit = 0
		elif limit > len(faces) - skip:
				limit = len(faces) - skip
		faces = faces[skip: skip + limit]
		return faces

	def delete_face_by_id(self, person_id: str, face_id: str):
		if not self.db_instance.check_face_by_id(person_id, face_id):
			raise HTTPException(status.HTTP_404_NOT_FOUND)
		collection = self.db_instance.get_person_collection()
		collection.update_one(
			{"id": person_id, "faces.id": face_id},
			{"$pull": {"faces": {"id": face_id}}}
		)
		image_path = os.path.join(
			self.database_config["path"],
			person_id, face_id + self.database_config["end"]
		)
		if os.path.exists(image_path):
				os.remove(image_path)

	def delete_all_face(self, person_id: str):
		if not self.db_instance.check_person_by_id(person_id):
			raise HTTPException(status.HTTP_404_NOT_FOUND)
		collection = self.db_instance.get_person_collection()
		collection.update_one(
			{"id": person_id},
			{"$pull": {"faces": {}}}
		)
		image_dir = os.path.join(self.database_config["path"], person_id)
		if os.path.exists(image_dir):
			shutil.rmtree(image_dir)