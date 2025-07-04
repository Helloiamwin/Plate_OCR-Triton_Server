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
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import json
import time
import logging
from PIL import Image
import utility as utility
import predict_rec as predict_rec
import  predict_det as predict_det
import  predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read 
from ppocr.utils.logging import get_logger
from  utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop
logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        # self.text_detector = predict_det.TextDetector()
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.drop_score = args.drop_score

        self.args = args
        self.crop_image_res_index = 0

    def __call__(self, img, cls=True):
        ori_im = img.copy()
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # dt_boxes = self.text_detector(img)
        rec_res = self.text_recognizer(img, "dt_boxes")

        # filter_boxes, filter_rec_res = [], []
        # for box, rec_result in zip(dt_boxes, rec_res):
        #     text, score = rec_result
        #     if score >= self.drop_score:
        #         filter_boxes.append(box)
        #         filter_rec_res.append(rec_result)
        # return filter_boxes, filter_rec_res

def main(args):
    text_sys = TextSystem(args)
    if args.image_dir:
        img = cv2.imread(args.image_dir)
    else:
        img = cv2.imread("img_11.jpg")
        
    text_sys(img)


if __name__ == "__main__":
    args = utility.parse_args()
    main(args)
