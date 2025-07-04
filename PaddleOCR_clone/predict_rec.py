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
from PIL import Image
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import math
import time
import traceback
import paddle
import copy
import  utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read
from  utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop
import tritonclient.grpc as grpcclient
logger = get_logger()

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

class TextRecognizer(object):
    def __init__(self, args):
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char
        }
        
        self.postprocess_op = build_post_process(postprocess_params)
        # self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(args, 'rec', logger)

        self.dictionary = (open(args.rec_char_dict_path, "r").read().split("\n"))
        self.dectionary_map = {x: i for i, x in enumerate(self.dictionary)}
        # print(self.dectionary_map)
        # print( self.dictionary)


        url = "localhost:8001"
        verbose = False
        # self.model_name = "pp_infer_text_det"
        self.triton_client = grpcclient.InferenceServerClient(url=url,verbose=verbose)

        self.max_length = 50

    def encode(self, text):
        encoded_text =[-1]*self.max_length
        for i,ch in enumerate(text):
            encoded_text[i] = self.dectionary_map[ch]
            if i == self.max_length -1:
                break
        encoded_text = np.array(encoded_text)
        return encoded_text

    def decode(self, encoded_text):
        text = ""
        for item in encoded_text:
            if item == -1: continue
            ch = self.dictionary[int(item)]
            text += ch
        return text

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
    
    def __call__(self,img, dt_boxes):

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
        print(dt_boxes.shape)
        print(dt_boxes)
        
        
        