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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import time
import sys
import utility as utility
import tritonclient.grpc as grpcclient

class TextDetector(object):
    def __init__(self):
        url = "localhost:8001"
        verbose = False
        self.model_name = "pp_infer_text_det"
        self.triton_client = grpcclient.InferenceServerClient(url=url,verbose=verbose)

    def __call__(self, imgs):
        inputs = []
        outputs = []
        bz,c,h,w = imgs.shape
        inputs.append(grpcclient.InferInput('text_det_input', [bz,3,h,w], "UINT8"))
        outputs.append(grpcclient.InferRequestedOutput('text_det_output'))

        inputs[0].set_data_from_numpy(imgs)
        results = self.triton_client.infer(model_name="pp_text_det",inputs=inputs, outputs=outputs)
        
        if results is None:
            raise ValueError("Inference failed, please check the model and inputs.")
        
        text_det_output = results.as_numpy('text_det_output').astype(np.float32)
        return text_det_output


if __name__ == "__main__":
    import copy
    text_detector = TextDetector()

    img1 = cv2.imread("11.jpg")
    img1 = cv2.resize(img1, (640,640))

    img2 = cv2.imread("00207393.jpg")
    img2 = cv2.resize(img2, (640,640))
    imgs = [img1, img2]
    ori_imgs = copy.deepcopy(imgs)
    imgs = np.array(imgs)
    imgs  = np.transpose(imgs, (0, 3, 1, 2))

    dt_boxes= text_detector(imgs)
    print(dt_boxes.shape)
    for i,dt_bbox in enumerate (dt_boxes):
        src_im = utility.draw_text_det_res(dt_boxes[i], ori_imgs[i])
        cv2.imwrite(f"result_{i}.jpg", src_im)