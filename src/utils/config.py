import time
import os
import yaml
from src.utils.singleton import Singleton
from src.utils.logger import Logger
class Config(object, metaclass=Singleton):
    def __init__(self) -> None:
        self.config = "configs/face_recognition.yaml"
        self.face_detection = None
        self.face_encode = None
        self.data_base = None
        self.faiss = None
        self.logger =  Logger().get_logger()
        self.init_config()

    def init_config(self):
        with open(self.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        face_detection_config = config["face_detection"]
        self.engine_name = face_detection_config["engine_name"]

        if self.engine_name in face_detection_config.keys():
            with open(face_detection_config[self.engine_name], 'r') as f:
                self.face_detection = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.logger.error(f"Does not support {self.engine_name} engine name")
            exit()

        with open(config["face_encode"], 'r') as f:
            self.face_encode = yaml.load(f, Loader=yaml.FullLoader)

        with open(config["database"], 'r') as f:
            self.data_base = yaml.load(f, Loader=yaml.FullLoader)

        with open(config["faiss"], 'r') as f:
            self.faiss = yaml.load(f, Loader=yaml.FullLoader)

    def get_face_detection(self):
        return self.face_detection
    
    def get_face_encode(self):
        return self.face_encode
    
    def get_database(self):
        return self.data_base
    
    def get_faiss(self):
        return self.faiss
    
    def get_detection_engine_name(self):
        return self.engine_name





