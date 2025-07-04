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
import matplotlib.pyplot as plt

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '/')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import numpy as np
import time
# from PaddleOCR_clone.ppocr.utils.logging import get_logger
import tritonclient.grpc as grpcclient
# logger = get_logger()

from unwanted_models.pp_pre_text_det_v2.first.ppocr.data import create_operators, transform

class VehicleDetection(object):
    def __init__(self, args):

        url = "localhost:8001"
        self.triton_client = grpcclient.InferenceServerClient(url=url,verbose=False)
        self.max_length = 50
        self.bouding_box = None
        self.vehicle_det_order_output = None

    def __call__(self, img_list, img_shape_list):

        inputs = []
        outputs = []

        self.img_list = img_list

        inputs.append(grpcclient.InferInput('vehicle_det_input', [*self.img_list.shape], "UINT8")) #(bz, c, h, w)
        inputs.append(grpcclient.InferInput('vehicle_det_input_shape_list', [self.img_list.shape[0], 2], "FP32"))

        outputs.append(grpcclient.InferRequestedOutput('vehicle_det_output'))
        # outputs.append(grpcclient.InferRequestedOutput('vehicle_det_order_output'))

        inputs[0].set_data_from_numpy(self.img_list)
        inputs[1].set_data_from_numpy(img_shape_list)

        t1 = time.time()
        results = self.triton_client.infer(model_name="yolo_vehicle_det_v2",inputs=inputs, outputs=outputs)
        print("infer time:", time.time() - t1)
        
        self.bouding_box = results.as_numpy('vehicle_det_output').astype(np.float32)
        
        # print("Shape of bouding box: ",self.bouding_box.shape)
        # print("Bounding boxes: ", self.bouding_box)
        # print("The order output result: ", self.vehicle_det_order_output)
        
    def test_(self, img_path_list):
        ###############
        imgs = [cv2.imread(img_path_list[i]) for i in range(len(img_path_list))]
        
        imgs_new = []
        for index in range(self.bouding_box.shape[0]):
            for box in self.bouding_box[index]:
                if box[0] != -1:
                    img_ = cv2.resize(imgs[index], (640, 640))
                    imgs_new.append(img_[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])])
            
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
        preprocess_op = create_operators(pre_process_list)
            
        # imgs_ = imgs_new # np.transpose(imgs_new,(0,2,3,1))
        results = []
        result_shape_list = []
        for img in imgs_new:
            img = img.astype(np.uint8).copy()
            
            shape_img = img.shape[:2]
            # results.append(cv2.resize(img, (640, 640)))
            
            data = {'image': cv2.resize(img, (640, 640))}
            data = transform(data, preprocess_op)
            img, _ = data     
            results.append(img)
                
            result_shape_list.append(shape_img)

        results = np.array(results)
        result_shape_list = np.array(result_shape_list)

        results = np.ascontiguousarray(results, dtype=np.float32) # (bz, c, h, w)
        result_shape_list = np.ascontiguousarray(result_shape_list, dtype=np.float32) # (bz, 2)
        
    def test_yolo_and_det(self):
        
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('text_det_input', [*self.img_list.shape], "UINT8")) #[3,-1,-1]
        inputs.append(grpcclient.InferInput('text_det_input_boxes', [*self.bouding_box.shape], "FP32")) #[-1,  4]

        outputs.append(grpcclient.InferRequestedOutput('text_det_output'))
        # outputs.append(grpcclient.InferRequestedOutput('vehicle_det_order_output'))

        inputs[0].set_data_from_numpy(self.img_list)
        inputs[1].set_data_from_numpy(self.bouding_box)

        t1 = time.time()
        results = self.triton_client.infer(model_name="pp_text_det_v2",inputs=inputs, outputs=outputs)
        print("infer time:", time.time() - t1)
        text_det_output = results.as_numpy('text_det_output').astype(np.float32) #[-1,4,2]
        
        img_cropped = []
        for index_img in range (self.img_list.shape[0]):
            for index_box, box in enumerate(self.bouding_box[index_img]):
                if box[0] != -1:
                    img_ =self.img_list[index_img]
                    img_ = np.transpose(img_, (1, 2, 0))  # Convert from CHW to HWC
                    cropped_img = img_[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]
                    cropped_img = cv2.resize(cropped_img, (640, 640))
                    img_cropped.append(cropped_img)
    
        imgs = []
        for index, boxes in enumerate(text_det_output):
            if boxes[0][0][0] == -1: continue 
            
            for box in boxes:
                if box[0][0] != -1:
                    img_ = img_cropped[index]
                    cv2.polylines(img_, [np.array(box, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            imgs.append(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))

        # Show images in a grid
        cols = 6
        rows = (len(imgs) + cols - 1) // cols
        plt.figure(figsize=(25, 5 * rows))
        for i, img in enumerate(imgs):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
        pass    
    
    def test_yolo_det_rec(self):
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('text_det_input', [*self.img_list.shape], "UINT8")) #[3,-1,-1]
        inputs.append(grpcclient.InferInput('text_det_input_boxes', [*self.bouding_box.shape], "FP32")) #[-1,  4]

        outputs.append(grpcclient.InferRequestedOutput('text_det_output'))

        inputs[0].set_data_from_numpy(self.img_list)
        inputs[1].set_data_from_numpy(self.bouding_box)

        t1 = time.time()
        results = self.triton_client.infer(model_name="pp_text_det_v2",inputs=inputs, outputs=outputs)
        print("infer time:", time.time() - t1)
        text_det_output = results.as_numpy('text_det_output').astype(np.float32) #[bz, -1,4,2]
        
        ########################
        
        inputs2 = []
        outputs2 = []
        inputs2.append(grpcclient.InferInput('vehicle_det_input', [*self.img_list.shape], "UINT8")) #[3,-1,-1]
        inputs2.append(grpcclient.InferInput('vehicle_det_output', [*self.bouding_box.shape], "FP32")) #[-1,  4]

        outputs2.append(grpcclient.InferRequestedOutput('pre_text_rec_input_images'))

        inputs2[0].set_data_from_numpy(self.img_list)
        inputs2[1].set_data_from_numpy(self.bouding_box)

        t1 = time.time()
        results2 = self.triton_client.infer(model_name="pp_pre_pre_text_rec",inputs=inputs2, outputs=outputs2)
        print("infer time:", time.time() - t1)
        text_det_output2 = results2.as_numpy('pre_text_rec_input_images').astype(np.uint8) #[bz, 3, -1, -1]
        
        ########################
        
        inputs3 = []
        outputs3 = []
        inputs3.append(grpcclient.InferInput('pre_text_rec_input_images', [*text_det_output2.shape], "UINT8")) #[bz, 3,-1,-1]
        inputs3.append(grpcclient.InferInput('pre_text_det_input_dt_boxes', [*text_det_output.shape], "FP32")) #[bz, -1,  4, 2]

        outputs3.append(grpcclient.InferRequestedOutput('pre_text_rec_output_order'))
        outputs3.append(grpcclient.InferRequestedOutput('pre_text_rec_output_images'))

        inputs3[0].set_data_from_numpy(text_det_output2)
        inputs3[1].set_data_from_numpy(text_det_output)

        t1 = time.time()
        results3 = self.triton_client.infer(model_name="pp_pre_text_rec_v2",inputs=inputs3, outputs=outputs3)
        print("infer time:", time.time() - t1)
        text_det_output3 = results3.as_numpy('pre_text_rec_output_images').astype(np.float32) #[bz, 3, 48, -1]
        text_det_output3_2 = results3.as_numpy('pre_text_rec_output_order').astype(np.int8) #[bz, -1]
        
        ########################
        
        inputs4 = []
        outputs4 = []
        inputs4.append(grpcclient.InferInput('x', [*text_det_output3.shape], "FP32")) #[bz, 3, 48, -1]

        outputs4.append(grpcclient.InferRequestedOutput('softmax_2.tmp_0'))

        inputs4[0].set_data_from_numpy(text_det_output3)

        t1 = time.time()
        results4 = self.triton_client.infer(model_name="pp_infer_text_rec",inputs=inputs4, outputs=outputs4)
        print("infer time:", time.time() - t1)
        text_det_output4 = results4.as_numpy('softmax_2.tmp_0').astype(np.float32) #[bz, -1, 97]
        
        ########################
        
        inputs5 = []
        outputs5 = []
        inputs5.append(grpcclient.InferInput('post_text_rec_input', [*text_det_output4.shape], "FP32")) #[bz, 3, 48, -1]

        outputs5.append(grpcclient.InferRequestedOutput('post_text_rec_output'))
        outputs5.append(grpcclient.InferRequestedOutput('post_text_rec_output_score'))

        inputs5[0].set_data_from_numpy(text_det_output4)

        t1 = time.time()
        results5 = self.triton_client.infer(model_name="pp_post_text_rec",inputs=inputs5, outputs=outputs5)
        print("infer time:", time.time() - t1)
        text_det_output5= results5.as_numpy('post_text_rec_output').astype(np.float32) #[bz**, -1, 50]
        text_det_output5_2= results5.as_numpy('post_text_rec_output_score').astype(np.float32) #[bz**, -1]
        
        ################
        def decode(encoded_text):
            dictionary = (open("/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/triton_models_new/pp_post_text_rec/1/en_dict.txt", "r").read().split("\n"))
            text = ""
            for item in encoded_text:
                if item == -1: continue
                ch = dictionary[int(item)]
                text += ch
            return text
        
        for text, score in zip(text_det_output5, text_det_output5_2):
            encoded_text = decode(text)
            print(encoded_text, "has score:", score)
        pass

    def draw_boxes_on_images(self, img_path_list, bouding_box, vehicle_det_order_output):
        imgs = []
        for i in range(len(img_path_list)):
            img = cv2.imread(img_path_list[i])
            start_index = sum(vehicle_det_order_output[:i])
            end_index = start_index + vehicle_det_order_output[i]
            boxes = bouding_box[start_index:end_index]
            if len(boxes) != 0:
                for box in boxes:
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0]) + int(box[2]), int(box[1]) + int(box[3])), (0,255,0), 2)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib

        # Show images in a grid
        cols = 3
        rows = (len(imgs) + cols - 1) // cols
        plt.figure(figsize=(25, 5 * rows))
        for i, img in enumerate(imgs):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
         
def test_vehicle_detection():
    IMAGE_PATH = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/images/car1.png'
    IMAGE_PATH_2 = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/PaddleOCR_clone/00207393.jpg'
    IMAGE_PATH_3 = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/YOLOv11/46844597805_0c180c2ebd_b.jpg'
    IMAGE_PATH_4 = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/images/car2.png'
    IMAGE_PATH_5 = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/images/car3.png'
    IMAGE_PATH_6 = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/images/car4.png'
    INPUT_SIZE = (640, 640) 
    
    class Args:
        image_dir_list = [IMAGE_PATH_2, IMAGE_PATH, IMAGE_PATH_3, IMAGE_PATH_4, IMAGE_PATH_5, IMAGE_PATH_6]
        
    args = Args()
    
    text_recognizer = VehicleDetection(args)

    img_list = []
    img_shape_list = []
    for img_dir in args.image_dir_list:
        img = cv2.imread(img_dir)
        img_shape_list.append(img.shape[:2])
        ori_im = img.copy()
        
        img = cv2.resize(img, INPUT_SIZE)
        img = np.transpose(img, (2, 0, 1))
        
        img_list.append(img)
        
    img_list = np.array(img_list)
    img_list = np.ascontiguousarray(img_list, dtype=np.uint8)
    
    img_shape_list = np.array(img_shape_list)
    img_shape_list = np.ascontiguousarray(img_shape_list, dtype=np.float32)
        
    text_recognizer(img_list, img_shape_list)
    # text_recognizer.draw_boxes_on_images(args.image_dir_list, text_recognizer.bouding_box, text_recognizer.vehicle_det_order_output)
    
    # text_recognizer.test_(args.image_dir_list)
    text_recognizer.test_yolo_and_det()
    # text_recognizer.test_yolo_det_rec()

def test_anything():
    # This function is a placeholder for any additional tests you might want to run.

    url = "localhost:8001"
    triton_client = grpcclient.InferenceServerClient(url=url,verbose=False)

    inputs = []
    outputs = []
    
    input_ = np.random.randn(2, 3, 48, 100).astype(np.float32)  # Example input shape (batch_size, channels, height, width)

    inputs.append(grpcclient.InferInput('x', [*input_.shape], "FP32")) #(bz, c, h, w)
    # inputs.append(grpcclient.InferInput('vehicle_det_input_shape_list', [img_list.shape[0], 2], "FP32"))

    # outputs.append(grpcclient.InferRequestedOutput('vehicle_det_output'))
    outputs.append(grpcclient.InferRequestedOutput('softmax_2.tmp_0'))

    inputs[0].set_data_from_numpy(input_)
    # inputs[1].set_data_from_numpy(img_shape_list)

    t1 = time.time()
    results = triton_client.infer(model_name="pp_infer_text_rec", inputs=inputs, outputs=outputs)
    print("infer time:", time.time() - t1)
    
    rec_result_ = results.as_numpy('softmax_2.tmp_0').astype(np.float32)
    
    postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": "/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/triton_models/pp_post_text_rec/1/en_dict.txt",
            "use_space_char": True
        }
    postprocess_op = build_post_process(postprocess_params)
    rec_result = postprocess_op(rec_result_)
    texts = [result[0] for result in rec_result]
    scores = [result[1] for result in rec_result]
    
    return_texts = []
    return_scores = []
    
    def encode(text):
        dictionary = (open("/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/triton_models/pp_post_text_rec/1/en_dict.txt", "r").read().split("\n"))
        dectionary_map = {x: i for i, x in enumerate(dictionary)}
        encoded_text =[-1]*50
        for i,ch in enumerate(text):
            encoded_text[i] = dectionary_map[ch]
            if i == 50 - 1:
                break
        encoded_text = np.array(encoded_text)
        return encoded_text
    
    for text, score in zip(texts, scores):
        encoded_text = encode(text)
        return_texts.append(encoded_text)
        return_scores.append(score)

    return_texts = np.ascontiguousarray(return_texts)
    return_scores = np.ascontiguousarray(return_scores)
    
    pass
            
if __name__ == "__main__":
    test_vehicle_detection()
    # test_anything()
    
    
    

