import cv2
import numpy as np 
import time
import yaml
import onnxruntime as ort
import copy
import torch
from src.models.yolov5.yolov5_utils import (letterbox,scale_boxes, non_max_suppression)
from src.utils.logger import Logger


class YOLOV5Onnx:
    def __init__(self, config: yaml) -> None:

        # Load config
        self.config = config
        self.onnx = self.config["onnx"]
        self.size = self.config["size"]
        self.batch_szie = self.config["batch_szie"]
        self.stride = self.config["stride"]
        self.device = self.config["device"]
        self.confidence_threshold = self.config["confidence_threshold"]
        self.iou_threshold = self.config["iou_threshold"]
        self.logger = Logger.__call__().get_logger()

        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif self.device =="gpu":
            providers = ["CUDAExecutionProvider"]
        else:
            self.logger.error(f"Does not support {self.device}")
            exit(0)

        # Init model
        self.sess = ort.InferenceSession(self.onnx, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = [self.sess.get_outputs()[0].name]

        self.logger.info("Initialize YOLOV5 Onnx successfully...")


    def pre_process(self, images: list):
        imgs = []
        for img in images:
            img = letterbox(img, self.size, stride=self.stride, auto=False)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = img / 255
            imgs.append(img)
        imgs = np.array(imgs).astype(np.float32)

        return imgs

    def post_process(self, pred: np.array, images, ori_images):
        pred = [torch.from_numpy(pred)]
        pred = non_max_suppression(pred, self.confidence_threshold, self.iou_threshold, None, False, 1000)

        post_det = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(images.shape[2:], det[:, :4], ori_images[i].shape).round()
                det = det.detach().cpu().numpy()
                post_det.append(det)

        return post_det

    def inference(self, images: list):
        if not isinstance(images, list):
            images = [images]
        ori_images = copy.deepcopy(images)
        images =  self.pre_process(images)

        pred = self.sess.run(self.output_name, {self.input_name: images})[0]

        pred = self.post_process(pred, images, ori_images)

        return pred

