import triton_python_backend_utils as pb_utils
import cv2
import json
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_config_0 = pb_utils.get_output_config_by_name(self.model_config, "pre_vehicle_det_output_images")
        self.output_dtype_0 = pb_utils.triton_string_to_numpy(output_config_0["data_type"])

        self.INPUT_SIZE = (640, 640)  # Default YOLO input size
        
        print('Initialized...')

    def execute(self, requests):
        print("Starting execution of pre_vehicle_det model...")
        responses = []
        for request in requests:
            imgs = pb_utils.get_input_tensor_by_name(request, "pre_vehicle_det_input_images").as_numpy()  # (bz, c, h, w)
            imgs = np.transpose(imgs,(0,2,3,1))
            results = []
            for img in imgs:
                img = img.astype(np.uint8).copy()
                # original_shape = img.shape[:2]
                
                resized = cv2.resize(img, self.INPUT_SIZE)
                
                resized = resized[:, :, ::-1]  # BGR to RGB
                resized = resized.transpose(2, 0, 1)  # HWC to CHW
                resized = resized.astype(np.float32) / 255.0
            
                results.append(resized)

            results = np.ascontiguousarray(results, dtype=self.output_dtype_0)
            out_tensor_0 = pb_utils.Tensor("pre_vehicle_det_output_images", results)

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)

        print("Finished execution of pre_vehicle_det model.")
        return responses

    def finalize(self):
        print('Cleaning up...')