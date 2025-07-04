import triton_python_backend_utils as pb_utils
import cv2
import json
import numpy as np
import math
import copy
# from ppocr.data import create_operators, transform

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(self.model_config, "pre_text_rec_input_images")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        
        print('Initialized...')
        
    def execute(self, requests):
        responses = []
        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "vehicle_det_input").as_numpy()
            dt_boxes = pb_utils.get_input_tensor_by_name(request, "vehicle_det_output").as_numpy()
            
            img_cropped = []
            for index_img in range (images.shape[0]):
                for index_box, box in enumerate(dt_boxes[index_img]):
                    if box[0] != -1:
                        img_ =images[index_img]
                        img_ = np.transpose(img_, (1, 2, 0))  # Convert from CHW to HWC
                        cropped_img = img_[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]
                        cropped_img = cv2.resize(cropped_img, (640, 640))
                        img_cropped.append(np.transpose(cropped_img, (2, 0, 1)))  # Convert back to CHW format
            
            img_cropped = np.ascontiguousarray(np.array(img_cropped), dtype=self.output_dtype)
            out_tensor = pb_utils.Tensor("pre_text_rec_input_images", img_cropped)
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')