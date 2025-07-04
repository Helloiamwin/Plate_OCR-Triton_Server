import triton_python_backend_utils as pb_utils
import cv2
import json
import numpy as np
from ppocr.postprocess import build_post_process
import logging

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(self.model_config, "post_text_det_output")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        postprocess_params = {}
        postprocess_params['name'] = 'DBPostProcess'
        postprocess_params["thresh"] = 0.5
        postprocess_params["box_thresh"] = 0.6
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = 1.5
        postprocess_params["use_dilation"] = False
        postprocess_params["score_mode"] = "fast"
        postprocess_params["box_type"] = 'quad'
        self.postprocess_op = build_post_process(postprocess_params)
        
        print('Initialized...')

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes
    
    def execute(self, requests):
        responses = []
        for request in requests:
            post_text_det_inputs = pb_utils.get_input_tensor_by_name(request, "post_text_det_input_images").as_numpy()
            # post_shape_list_inputs = pb_utils.get_input_tensor_by_name(request, "post_text_det_input_shape_list").as_numpy()

            post_shape_list_inputs = np.full((post_text_det_inputs.shape[0], 4), [640, 640, 1, 1], dtype=np.float32)

            batch_size = post_text_det_inputs.shape[0]
            
            # logging.warning(f"-------------Batch size: {batch_size}, Input shape: {post_text_det_inputs.shape}, Shape list: {post_shape_list_inputs.shape}")

            preds = {}
            preds["maps"] = post_text_det_inputs
            post_result = self.postprocess_op(preds, post_shape_list_inputs)
            tmp_results = []

            for i in range(batch_size):
                dt_boxes = post_result[i]["points"]
                img_shape = (post_shape_list_inputs[i][0], post_shape_list_inputs[i][1], 3)
                dt_boxes = self.filter_tag_det_res(dt_boxes, img_shape)
                tmp_results.append(dt_boxes)

            # logging.warning(f"---------Postprocess results: {len(tmp_results)} boxes detected in batch, {tmp_results}")

            max_shape = max([item.shape[0] for item in tmp_results])
            results = []
            for tmp_result in tmp_results:
                extra = max_shape - tmp_result.shape[0]
                if extra != 0:
                    dummy_array = np.ones((extra, 4, 2), dtype=np.float32)*-1
                    if tmp_result.shape[0] == 0:
                        result = np.ones((extra, 4, 2), dtype=np.float32)*-1
                    else:
                        result =  np.concatenate((tmp_result, dummy_array), axis=0)
                else:
                    result = tmp_result
                results.append(result)

            results = np.ascontiguousarray(results, dtype=self.output0_dtype)

            out_tensor_0 = pb_utils.Tensor("post_text_det_output", results.astype(self.output0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Cleaning up...')
        
        