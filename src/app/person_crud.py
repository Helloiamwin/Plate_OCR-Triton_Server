from fastapi import status, HTTPException
from src.database.person_db import PersonDatabase
from src.utils.config import Config
from pathlib import Path
import shutil
import os
import copy 

class PersonCRUD:
	def __init__(self) -> None:
		self.database_config = Config.__call__().get_database()
		self.db_instance = PersonDatabase.__call__()

	def insert_person(self, id, name):
		collection = self.db_instance.get_person_collection()
		if self.db_instance.check_person_by_id(id):
			raise HTTPException(status.HTTP_409_CONFLICT)
		person = {"id": id, "name": name}
		collection.insert_one(copy.deepcopy(person))
		return person

	def select_all_people(self, skip: int, limit: int):
		list_people = []
		collection = self.db_instance.get_person_collection()
		docs = collection.find(
				{}, {"_id": 0, "id": 1, "name":1}).skip(skip).limit(limit)
		for doc in docs:
			list_people.append(doc)
		return list_people
	
	def select_person_by_id(self, person_id):
		collection = self.db_instance.get_person_collection()
		if not self.db_instance.check_person_by_id(person_id):
			raise HTTPException(status.HTTP_404_NOT_FOUND)
		doc = collection.find_one(
			{"id": person_id},{"_id": 0, "id": 1, "name":1})
		return doc
	
	def update_person_name(self, person_id: str, name: str):
		collection = self.db_instance.get_person_collection()
		if not self.db_instance.check_person_by_id(person_id):
			raise HTTPException(status.HTTP_404_NOT_FOUND)
		collection.update_one({"id": person_id},{"$set": {"name": name}})

	def delete_person_by_id(self, id: str):
		collection = self.db_instance.get_person_collection()
		if not self.db_instance.check_person_by_id(id):
			raise HTTPException(status.HTTP_404_NOT_FOUND)
		image_dir = os.path.join(self.database_config["path"], id)
		if os.path.exists(image_dir):
			shutil.rmtree(image_dir)
		collection.delete_one({"id": id})
	
	def delete_all_people(self):
		collection = self.db_instance.get_person_collection()
		collection.delete_many({})
		if os.path.exists(self.database_config["path"]):
			shutil.rmtree(self.database_config["path"])
			Path(self.database_config["path"]).mkdir(
				parents=True, exist_ok=True)