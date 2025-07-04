import triton_python_backend_utils as pb_utils
import cv2
import json
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(self.model_config, "post_vehicle_det_output")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        self.INPUT_SIZE = [640, 640]
        self.CONFIDENCE_THRESHOLD = 0.3
        self.NMS_THRESHOLD = 0.4
        
        print('Initialized...')

    def execute(self, requests):
        print("Starting execution of post_vehicle_det model...")
        responses = []
        for request in requests:
            post_vehicle_det_input = pb_utils.get_input_tensor_by_name(request, "post_vehicle_det_input_images").as_numpy()

            print(f"Input shape: {post_vehicle_det_input.shape}, dtype: {post_vehicle_det_input.dtype}")

            results = []
            order_results = []
            for index, output in enumerate(post_vehicle_det_input):
                predictions = np.squeeze(output)
                predictions = predictions.T

                boxes = []
                confidences = []
                class_ids = []

                for pred in predictions:
                    x, y, w, h = pred[0:4]
                    class_scores = pred[4:]
                    class_id = np.argmax(class_scores)
                    confidence = class_scores[class_id]

                    if confidence > self.CONFIDENCE_THRESHOLD and class_id == 2:
                        x = int(x - w / 2)
                        y = int(y - h / 2) 
                        width = int(w)
                        height = int(h)

                        boxes.append([x, y, width, height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

                    indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)

                    # indices = np.array(indices).flatten()
                    pred_output = [boxes[i] for i in indices]

                results.append(pred_output)
                order_results.append(len(pred_output))
                
                max_order = max(order_results) if order_results else 0
                for index in range(len(results)):
                    if len(results[index]) < max_order:
                        results[index] += np.full((max_order - len(results[index]), 4), -1).tolist()
                        
            # Ensure results is always a 2D array with shape (N, 4)
            results = np.array(results)
            results = np.ascontiguousarray(results, dtype=self.output0_dtype)
            
            out_tensor_0 = pb_utils.Tensor("post_vehicle_det_output", results.astype(self.output0_dtype))
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')