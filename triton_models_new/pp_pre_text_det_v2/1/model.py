import triton_python_backend_utils as pb_utils
import cv2
import json
import numpy as np
from ppocr.data import create_operators, transform

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_config_0 = pb_utils.get_output_config_by_name(self.model_config, "pre_text_det_output_images")
        self.output_dtype_0 = pb_utils.triton_string_to_numpy(output_config_0["data_type"])

        # output_config_1 = pb_utils.get_output_config_by_name(self.model_config, "pre_text_det_output_shape_list")
        # self.output_dtype_1 = pb_utils.triton_string_to_numpy(output_config_1["data_type"])

        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'limit_type': 'max',
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]

        self.preprocess_op = create_operators(pre_process_list)

        print('Initialized...')

    def execute(self, requests):
        responses = []
        for request in requests:
            imgs = pb_utils.get_input_tensor_by_name(request, "pre_text_det_input_images").as_numpy()  # (bz, c, h, w)
            boxes = pb_utils.get_input_tensor_by_name(request, "pre_text_det_input_boxes").as_numpy ()  # (bz, -1, 4)
            
            imgs_new = []
            imgs = np.transpose(imgs,(0,2,3,1))
            # crop images based on boxes
            for index in range(imgs.shape[0]):
                for box in boxes[index]:
                    if box[0] != -1:
                        img_ = cv2.resize(imgs[index], (640, 640))
                        imgs_new.append(img_[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])])
                
            # imgs = np.transpose(imgs_new,(0,2,3,1))
            results = []
            for img in imgs_new:
                img = img.astype(np.uint8).copy()
                data = {'image': cv2.resize(img, (640, 640))}
                data = transform(data, self.preprocess_op)
                img, _ = data

                results.append(img)

            results = np.array(results)
            results = np.ascontiguousarray(results, dtype=self.output_dtype_0)

            out_tensor_0 = pb_utils.Tensor("pre_text_det_output_images", results)

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')