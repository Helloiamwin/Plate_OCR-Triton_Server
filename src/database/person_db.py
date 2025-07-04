from src.database.database import BaseDatabase
from src.utils.singleton import Singleton
from src.utils.config import Config
import numpy as np

class PersonDatabase(BaseDatabase, metaclass=Singleton):
	def __init__(self):
		self.config = Config.__call__().get_database()
		super(PersonDatabase, self).__init__(self.config)

		database_name = self.config["database_name"]
		collection_name = self.config["collection_name"]
		self.database = self.client[database_name]
		self.person_collection = self.database[collection_name]

	def get_person_collection(self):
		return self.person_collection
	
	def initialize_local_db(self):
		person_docs = self.person_collection.find()
		self.vectors = {}
		self.people = {}
		for person in person_docs:
			person_id = person["id"]
			person_name = person["name"]
			if ("faces" not in person.keys()) or (person["faces"] is None):
				continue
			for face in person["faces"]:
				if ("vector" not in face.keys()) or (face["vector"] is None):
					continue
				vector_id = face["id"]
				vector = np.array(face["vector"], dtype=np.float32)
				self.vectors[vector_id] = vector / np.linalg.norm(vector)
				self.people[vector_id] = {
					"person_id": person_id,
					"person_name": person_name
				}

	def get_local_db(self):
		return self.people, self.vectors

	def remove_person_from_local_db(self, person_id: str):
		keys_remove = [
			vector_id for vector_id in self.people.keys() 
			if self.people[vector_id]["person_id"] == person_id
		]
		for key in keys_remove:
			del self.people[key]
			del self.vectors[key] 

	def remove_vector_from_local_db(self, vector_id: str):
		if vector_id in self.people.keys():
			del self.people[vector_id]
		if vector_id in self.vectors.keys():
			del self.vectors[vector_id]

	def update_name_to_local_db(self, person_id: str, name: str):
		for key in self.people.keys():
			if self.people[key]["person_id"] == person_id:
				self.people[key]["person_name"] = name

	def update_id_to_local_db(self, current_person_id: str, new_person_id: str):
		for key in self.people.keys():
			if self.people[key]["person_id"] == current_person_id:
				self.people[key]["person_id"] = new_person_id

	# def add_vector_to_local_db(self, person: PersonDoc):
	# 	person_id = person["id"]
	# 	person_name = person["name"]
	# 	for face in person["faces"]:
	# 		if ("vectors" not in face.keys()) or (len(face["vectors"]) == 0):
	# 			continue
	# 		for vector in face["vectors"]:
	# 			vector_id = vector["id"]
	# 			value = np.array(vector["value"], dtype=np.float32)
	# 			self.vectors[vector_id] = value / np.linalg.norm(value)
	# 			self.people[vector_id] = {
	# 				"person_id": person_id,
	# 				"person_name": person_name
	# 			}

	def update_vector(self, vector_id, new_value):
		value = np.array(new_value, dtype=np.float32)
		self.vectors[vector_id] = value / np.linalg.norm(value)
	
	def remove_all_db(self):
		self.people = {}
		self.vectors = {}

	######################################################################################
	def check_person_by_id(self, person_id: str) -> bool:
		person_doc =self.person_collection.find_one({"id": person_id}, {"_id": 0})
		if person_doc is None:
			return False
		return True
	
	def check_face_by_id(self, person_id: str, face_id: str) -> bool:
		person_doc = self.person_collection.find_one({"id": person_id}, {"_id": 0, "faces.vectors": 0})
		if person_doc is None:
			return False
		if "faces" not in person_doc.keys():
			return False
		elif person_doc["faces"] is None:
			return False
		elif len(person_doc["faces"]) == 0:
			return False
		current_ids = [x["id"] for x in person_doc["faces"]]
		if face_id not in current_ids:
			return False
		return True