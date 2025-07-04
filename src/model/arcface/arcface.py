from src.model.utils import face_align, face_detect
from src.utils.logger import Logger
from datetime import datetime
import numpy as np
import onnxruntime
import cv2

class ArcFace:
	def __init__(self, config) -> None:
		self.config = config
		self.input_size = self.config["input_size"]
		self.output_shape = self.config["output_shape"]
		self.device = self.config["device"]
		self.model_path = self.config["model_path"]
		self.input_std = self.config["input_std"]
		self.input_mean = self.config["input_mean"]
		self.output_names = self.config["output_names"]
		self.input_names = self.config["input_names"]
		self.threshold = self.config["threshold"]
		
		self.logger = Logger.__call__().get_logger()

		if self.device == "gpu":
			provider = "CUDAExecutionProvider"
		elif self.device == "cpu":
			provider = "CPUExecutionProvider"
		elif self.device == "tensorrt":
			provider = "TensorrtExecutionProvider"
		else:
			self.logger.error(f"Does not support {self.device}")
			exit(0)
		self.session = onnxruntime.InferenceSession(self.model_path, providers=[provider])
		self.logger.info("Initialize ARC FACE Onnx successfully...")
	
	def get(self, img: np.ndarray, face: face_detect.Face):
		aimg = face_align.norm_crop(img, landmark=face.kps)
		face.embedding = self.get_feat(aimg).flatten()
		return face.normed_embedding
	
	def get_feat(self, imgs: np.ndarray):
		if not isinstance(imgs, list):
			imgs = [imgs]
		input_size = self.input_size
		
		blob = cv2.dnn.blobFromImages(
			imgs, 1.0 / self.input_std, input_size,
			(self.input_mean, self.input_mean, self.input_mean), swapRB=True
		)
		
		net_out = self.session.run(self.output_names, {self.input_names[0]: blob})[0]
		return net_out
	