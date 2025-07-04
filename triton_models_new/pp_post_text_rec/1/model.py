import triton_python_backend_utils as pb_utils
import cv2
import json
import numpy as np
from ppocr.postprocess import build_post_process
import logging
import os

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_0_config = pb_utils.get_output_config_by_name(self.model_config, "post_text_rec_output")
        self.output_0_dtype = pb_utils.triton_string_to_numpy(output_0_config["data_type"])

        output_1_config = pb_utils.get_output_config_by_name(self.model_config, "post_text_rec_output_score")
        self.output_1_dtype = pb_utils.triton_string_to_numpy(output_1_config["data_type"])

        current_dir_path =  os.path.dirname(os.path.abspath(__file__))
        dictionary_file = os.path.join(current_dir_path, "en_dict.txt")
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": dictionary_file,
            "use_space_char": True
        }
        self.postprocess_op = build_post_process(postprocess_params)

        dictionary = (open(dictionary_file, "r").read().split("\n"))
        self.dectionary_map = {x: i for i, x in enumerate(dictionary)}

        print('Initialized...')

        self.max_length = 50

    def encode(self, text):
        encoded_text =[-1]*self.max_length
        for i,ch in enumerate(text):
            encoded_text[i] = self.dectionary_map[ch]
            if i == self.max_length - 1:
                break
        encoded_text = np.array(encoded_text)
        return encoded_text
    
    def execute(self, requests):
        responses = []
        for request in requests:
            rec_result = pb_utils.get_input_tensor_by_name(request, "post_text_rec_input").as_numpy()
            rec_result = self.postprocess_op(rec_result)
            texts = [result[0] for result in rec_result]
            scores = [result[1] for result in rec_result]

            return_texts = []
            return_scores = []
            for text, score in zip(texts, scores):
                encoded_text = self.encode(text)
                return_texts.append(encoded_text)
                return_scores.append(score)

            return_texts = np.ascontiguousarray(return_texts, dtype=self.output_0_dtype)
            return_scores = np.ascontiguousarray(return_scores, dtype=self.output_1_dtype)

            out_tensor_text = pb_utils.Tensor("post_text_rec_output", return_texts.astype(self.output_0_dtype))
            out_tensor_score = pb_utils.Tensor("post_text_rec_output_score", return_scores.astype(self.output_1_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_text, out_tensor_score])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Cleaning up...')
        