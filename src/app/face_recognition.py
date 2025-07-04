from src.model.scrfd.scrfd import SCRFD
from src.model.arcface.arcface import ArcFace
from src.model.utils.face_detect import Face
from src.model.faiss.faiss_wrap import FAISS
from src.manager.detection_manager import DetectionManager
from src.utils.config import Config
from src.utils.logger import Logger
from src.utils.singleton import Singleton
import numpy as np

class FaceRecognition(metaclass=Singleton):
    def __init__(self) -> None:
        self.config = Config()
        self.face_detection_config = self.config.get_face_detection()
        self.face_detection_engine_name = self.config.get_detection_engine_name()
        self.face_encode_config = self.config.get_face_encode()
        self.faiss_config = self.config.get_faiss()

        self.detection_manager = DetectionManager(self.face_detection_config, 
                                                self.face_detection_engine_name)
        self.face_detection =  self.detection_manager.get_engine()
        self.face_encode = ArcFace(self.face_encode_config)
        self.faiss = FAISS(self.faiss_config)

    def get_face_detection(self):
        return self.face_detection
    
    def get_face_encode(self):
        return self.face_encode
    
    def reload(self):
        self.faiss.reload()
    
    def recognize(self, image:np.array):
        dets, kps, encodes = self.get_encode(image)
        rec  = self.faiss.search(np.array(encodes), 5)
        results =  self.convert_result_to_dict(dets, kps, rec)
        return results

    def get_encode(self, image:np.array, largest_bbox = False) -> np.array:
        detection_results = self.face_detection.detect(image)
        if detection_results[0].shape[0] == 0 or detection_results[1].shape[0] == 0:
            return np.array([])
        if largest_bbox :
            detection_results = self.get_largest_bbox(detection_results)
        encode_results = []
        for i,detection_result in enumerate(detection_results[0]):
            face = Face(bbox=detection_result[:4], kps=detection_results[1][i], det_score=detection_result[4])
            encode_result = self.face_encode.get(image, face)
            encode_results.append(encode_result)
        results = [detection_results[0], detection_results[1], encode_results]
        
        return results
    
    def get_largest_bbox(self, detection_results):
        bboxes = detection_results[0]
        kpss = detection_results[1]
        largest_bbox = max(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        kps_corr = kpss[np.where((bboxes==largest_bbox).all(axis=1))]
        detection_largest = (np.expand_dims(largest_bbox, axis=0), kps_corr)

        return detection_largest
    
    def convert_result_to_dict(self, detections, landmarks, recognitions):
        recognition_results = []
        for i in range(len(detections)):
            face = {}
            bbox = detections[i][:4].tolist()
            detection_score = float(detections[i][4])
            landmark = []
            for point in landmarks[i]:
                landmark.append(point.tolist())
            face["bbox"] = bbox
            face["score"] = detection_score
            face["landmark"] = landmark
            face["recognition"] = recognitions[i]
            recognition_results.append(face)
        return recognition_results