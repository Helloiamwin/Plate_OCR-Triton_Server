
from src.model.utils.face_detect import nms, distance2bbox, distance2kps, custom_resize
from src.utils.logger import Logger
from datetime import datetime
import onnxruntime
import numpy as np
import cv2

class SCRFD:
	def __init__(self, config) -> None:
		self.config = config
		self.model_path = self.config["model_path"]
		self.input_size = self.config["input_size"]
		self.device = self.config["device"]
		self.input_std = self.config["input_std"]
		self.input_mean = self.config["input_mean"]
		self.nms_thresh = self.config["nms_thresh"]
		self.det_thresh = self.config["det_thresh"]
		self.num_anchors = self.config["num_anchors"]
		self.output_names = self.config["output_names"]
		self.input_names = self.config["input_names"]
		self.feat_stride_fpn = self.config["feat_stride_fpn"]
		self.fmc = self.config["fmc"]

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
		self.center_cache = {}

		self.logger.info("Initialize SCRFD Onnx successfully...")

	def forward(self, img: np.ndarray, threshold: float):
		scores_list = []
		bboxes_list = []
		kpss_list = []
		input_size = tuple(img.shape[0:2][::-1])
		blob = cv2.dnn.blobFromImage(
			img, 1.0/self.input_std, input_size, 
			(self.input_mean, self.input_mean, self.input_mean), 
			swapRB=True
		)
		
		

		import time 
		sum = 0
		for i in range(101):
			if i == 0: continue		
			st = time.time()
			net_outs = self.session.run(self.output_names, {self.input_names[0] : blob})
			end = time.time()
			sum += end-st
		print(sum/100*1000)


		input_height = blob.shape[2]
		input_width = blob.shape[3]
		for idx, stride in enumerate(self.feat_stride_fpn):
			scores = net_outs[idx][0]
			bbox_preds = net_outs[idx + self.fmc][0] * stride
			kps_preds = net_outs[idx + self.fmc * 2][0] * stride

			height = input_height // stride
			width = input_width // stride
			key = (height, width, stride)
			if key in self.center_cache:
				anchor_centers = self.center_cache[key]
			else:
				anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
				anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
				if self.num_anchors > 1:
					anchor_centers = np.stack([anchor_centers]*self.num_anchors, axis=1).reshape( (-1,2) )
				if len(self.center_cache) <  100:
					self.center_cache[key] = anchor_centers

			pos_inds = np.where(scores>=threshold)[0]
			bboxes = distance2bbox(anchor_centers, bbox_preds)
			pos_scores = scores[pos_inds]
			pos_bboxes = bboxes[pos_inds]
			scores_list.append(pos_scores)
			bboxes_list.append(pos_bboxes)
			kpss = distance2kps(anchor_centers, kps_preds)
			kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
			pos_kpss = kpss[pos_inds]
			kpss_list.append(pos_kpss)

		return scores_list, bboxes_list, kpss_list
	
	def detect(self, img, input_size = None, max_num=0, metric='default'):
		
		assert input_size is not None or self.input_size is not None
		input_size = self.input_size if input_size is None else input_size   
		det_img, det_scale = custom_resize(img, input_size)
		scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)
		scores = np.vstack(scores_list)
		scores_ravel = scores.ravel()
		order = scores_ravel.argsort()[::-1]
		bboxes = np.vstack(bboxes_list) / det_scale
		kpss = np.vstack(kpss_list) / det_scale
		pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
		pre_det = pre_det[order, :]
		keep = nms(pre_det, self.nms_thresh)
		det = pre_det[keep, :]
		kpss = kpss[order,:,:]
		kpss = kpss[keep,:,:]

		if max_num > 0 and det.shape[0] > max_num:
			area = (det[:, 2] - det[:, 0]) * (det[:, 3] -	det[:, 1])
			img_center = img.shape[0] // 2, img.shape[1] // 2
			offsets = np.vstack([
				(det[:, 0] + det[:, 2]) / 2 - img_center[1],
				(det[:, 1] + det[:, 3]) / 2 - img_center[0]
			])
			offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
			if metric=='max':
				values = area
			else:
				values = area - offset_dist_squared * 2.0  # some extra weight on the centering
			bindex = np.argsort(values)[::-1]  # some extra weight on the centering
			bindex = bindex[0:max_num]
			det = det[bindex, :]
			if kpss is not None:
				kpss = kpss[bindex, :]
		return det, kpss