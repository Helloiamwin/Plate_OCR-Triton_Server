import pymongo
from datetime import datetime
from src.utils.logger import Logger

class BaseDatabase(object):
	def __init__(self, config):
		self.hostname = config["hostname"]
		self.port = config["port"]
		self.user = config["user"]
		self.password = config["password"]
		if (self.user == None or self.password == None) or ((self.user == "" or self.password == "")):
			self.url = f"mongodb://{self.hostname}:{self.port}"
		else:
			self.url = f"mongodb://{self.user}:{self.password}@{self.hostname}:{self.port}"
		self.logger = Logger.__call__().get_logger()
		self.Initialize()

	def Initialize(self):
		try:
			self.client = pymongo.MongoClient(self.url)
			self.logger.info("Connected to MongoDB.")
		except pymongo.errors.ServerSelectionTimeoutError as err:
			self.logger.info(f"Cannot connect to MongoDB: {err}")