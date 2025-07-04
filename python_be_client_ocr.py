# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import cv2
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import numpy as np
import time
# from PaddleOCR_clone.ppocr.utils.logging import get_logger
import tritonclient.grpc as grpcclient
# logger = get_logger()

class TextRecognizer(object):
    def __init__(self, args):
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]

        self.dictionary = (open(args.rec_char_dict_path, "r").read().split("\n"))
        self.dectionary_map = {x: i for i, x in enumerate(self.dictionary)}
        # print(self.dectionary_map)
        # print( self.dictionary)

        url = "localhost:8001"
        verbose = False
        self.triton_client = grpcclient.InferenceServerClient(url=url,verbose=verbose)
        self.max_length = 50

    def decode(self, encoded_text):
        text = ""
        for item in encoded_text:
            if item == -1: continue
            ch = self.dictionary[int(item)]
            text += ch
        return text

    def __call__(self,img, dt_boxes = None):

        inputs = []
        outputs = []
        bz,c,h,w = img.shape
        inputs.append(grpcclient.InferInput('ocr_input', [bz,3,h,w], "UINT8"))

        outputs.append(grpcclient.InferRequestedOutput('ocr_output_encoded_text'))
        outputs.append(grpcclient.InferRequestedOutput('ocr_output_score'))
        outputs.append(grpcclient.InferRequestedOutput('ocr_output_dt_boxes'))

        inputs[0].set_data_from_numpy(img)

        t1 = time.time()
        results = self.triton_client.infer(model_name="ocr",inputs=inputs, outputs=outputs)
        print("infer time:", time.time() - t1)
        
        texts = results.as_numpy('ocr_output_encoded_text').astype(np.float32)
        scores = results.as_numpy('ocr_output_score').astype(np.float32)
        dt_boxes = results.as_numpy('ocr_output_dt_boxes').astype(np.float32)

        for encoded_text, score in zip(texts, scores):
            plain_text = self.decode(encoded_text)
            print(plain_text, score)
        print(dt_boxes.shape, scores.shape)
        # print(dt_boxes)
        
if __name__ == "__main__":
    class Args:
        image_dir = "./PaddleOCR_clone/IMG_3842-1-1024x1024.png"
        rec_char_dict_path = "./PaddleOCR_clone/ppocr/utils/en_dict.txt"
        use_angle_cls = False
        rec_image_shape = "3, 32, 100"

    args = Args()
    
    text_recognizer = TextRecognizer(args)

    img = cv2.imread(args.image_dir)
    ori_im = img.copy()
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    text_recognizer(img)





