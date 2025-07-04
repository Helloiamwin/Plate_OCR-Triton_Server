import triton_python_backend_utils as pb_utils
import cv2
import json
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(self.model_config, "post_vehicle_det_output")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        
        output_1_config = pb_utils.get_output_config_by_name(self.model_config, "post_vehicle_det_order_output")
        self.output_1_dtype = pb_utils.triton_string_to_numpy(output_1_config["data_type"])

        self.INPUT_SIZE = [640, 640]
        self.CONFIDENCE_THRESHOLD = 0.3
        self.NMS_THRESHOLD = 0.4
        
        print('Initialized...')

    def execute(self, requests):
        print("Starting execution of post_vehicle_det model...")
        responses = []
        for request in requests:
            post_vehicle_det_input = pb_utils.get_input_tensor_by_name(request, "post_vehicle_det_input_images").as_numpy()
            post_shape_list_inputs = pb_utils.get_input_tensor_by_name(request, "post_vehicle_det_input_shape_list").as_numpy()

            print(f"Input shape: {post_vehicle_det_input.shape}, dtype: {post_vehicle_det_input.dtype}")
            print(f"Input shape list: {post_shape_list_inputs.shape}, dtype: {post_shape_list_inputs.dtype}")

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
                        x = int((x - w / 2) * post_shape_list_inputs[index][1] / self.INPUT_SIZE[0])
                        y = int((y - h / 2) * post_shape_list_inputs[index][0] / self.INPUT_SIZE[1])
                        width = int(w * post_shape_list_inputs[index][1] / self.INPUT_SIZE[0])
                        height = int(h * post_shape_list_inputs[index][0] / self.INPUT_SIZE[1])

                        boxes.append([x, y, width, height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

                    indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)

                    # indices = np.array(indices).flatten()
                    pred_output = [boxes[i] for i in indices]

                results.extend(pred_output)
                order_results.append(len(pred_output))
                
            # Ensure results is always a 2D array with shape (N, 4)
            results = np.array(results)
            results = np.ascontiguousarray(results, dtype=self.output0_dtype)
            
            order_results = np.array(order_results)
            order_results = np.ascontiguousarray(order_results, dtype=self.output_1_dtype)
            
            out_tensor_0 = pb_utils.Tensor("post_vehicle_det_output", results.astype(self.output0_dtype))
            out_tensor_1 = pb_utils.Tensor("post_vehicle_det_order_output", order_results.astype(self.output_1_dtype))
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')