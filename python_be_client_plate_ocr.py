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


class Plate_OCR(object):
    def __init__(self, args):

        url = "localhost:8001"
        self.triton_client = grpcclient.InferenceServerClient(url=url,verbose=False)
        
        self.bb_vehicle = None
        self.bb_text_det = None
        self.order_text_rec = None
        self.output_text = None
        self.output_score = None

    def __call__(self, img_list, img_shape_list):

        inputs = []
        outputs = []

        self.img_list = img_list

        inputs.append(grpcclient.InferInput('plate_ocr_image_input', [*self.img_list.shape], "UINT8")) #(bz, c, h, w)
        inputs.append(grpcclient.InferInput('plate_ocr_input_shape_list', [self.img_list.shape[0], 2], "FP32")) #(bz, 2)

        outputs.append(grpcclient.InferRequestedOutput('vehicle_det_output')) # (bz, -1, 4)
        outputs.append(grpcclient.InferRequestedOutput('text_det_output')) # (bz*, -1, 4, 2)
        outputs.append(grpcclient.InferRequestedOutput('pre_text_rec_output_order')) # (bz*, -1)
        outputs.append(grpcclient.InferRequestedOutput('plate_ocr_output_encoded_text')) # (bz**, -1, 50)
        outputs.append(grpcclient.InferRequestedOutput('plate_ocr_output_score')) # (bz**, -1)

        inputs[0].set_data_from_numpy(self.img_list)
        inputs[1].set_data_from_numpy(img_shape_list)

        t1 = time.time()
        results = self.triton_client.infer(model_name="plate_ocr",inputs=inputs, outputs=outputs)
        print("Infer time:", time.time() - t1)
        
        self.bb_vehicle = results.as_numpy('vehicle_det_output').astype(np.float32)
        self.bb_text_det = results.as_numpy('text_det_output').astype(np.float32)
        self.order_text_rec = results.as_numpy('pre_text_rec_output_order').astype(np.float32)
        self.output_text = results.as_numpy('plate_ocr_output_encoded_text').astype(np.float32)
        self.output_score = results.as_numpy('plate_ocr_output_score').astype(np.float32)
        
        pass
    
    def show_text_recognition_results(self, img_path_list):
        imgs = []
        start_index_text_det = 0
        for id_img in range(len(img_path_list)):
            img = cv2.imread(img_path_list[id_img])
            img = cv2.resize(img, (640, 640))
            for box in self.bb_vehicle[id_img]:
                if box[0] != -1:
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0]) + int(box[2]), int(box[1]) + int(box[3])), (0,255,0), 2)
                    
                    start_index_text_det_det = 0
                    for text_box in self.bb_text_det[start_index_text_det]:
                        if text_box[0][0] != -1:
                            text_box[:, 0] = text_box[:, 0] / 640 * int(box[2]) + int(box[0])
                            text_box[:, 1] = text_box[:, 1] / 640 * int(box[3]) + int(box[1])
                            cv2.polylines(img, [np.array(text_box, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
                              
                            index_text = start_index_text_det * int(self.order_text_rec[0]) + start_index_text_det_det
                            
                            if self.output_score[index_text] > 0.5:
                                encoded_text = self.decode(self.output_text[index_text])
                                print(self.output_score[index_text], encoded_text)
                                cv2.putText(img, encoded_text, (int(text_box[0][0] - 10), int(text_box[0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                
                        start_index_text_det_det += 1
                    start_index_text_det += 1
                        
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
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

    def decode(self, encoded_text):
        dictionary = (open("/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/triton_models_new/pp_post_text_rec/1/en_dict.txt", "r").read().split("\n"))
        text = ""
        for item in encoded_text:
            if item == -1: continue
            ch = dictionary[int(item)]
            text += ch
        return text 
       
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
        
        for text, score in zip(text_det_output5, text_det_output5_2):
            encoded_text = self.decode(text)
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
         
def test_plate_detection():
    IMAGE_PATH = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/images/car5.png'
    IMAGE_PATH_2 = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/images/car6.png'
    IMAGE_PATH_3 = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/images/car7.png'
    IMAGE_PATH_4 = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/images/car8.png'
    IMAGE_PATH_5 = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/images/car9.png'
    IMAGE_PATH_6 = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/images/car10.png'
    INPUT_SIZE = (640, 640) 

    class Args:
        image_dir_list = [IMAGE_PATH, IMAGE_PATH_2, IMAGE_PATH_3, IMAGE_PATH_4, IMAGE_PATH_5, IMAGE_PATH_6] #, IMAGE_PATH_5, IMAGE_PATH_6]

    args = Args()

    text_recognizer = Plate_OCR(args)

    img_list = []
    img_shape_list = []
    for img_dir in args.image_dir_list:
        img = cv2.imread(img_dir)
        img_shape_list.append(img.shape[:2])
        
        img = cv2.resize(img, INPUT_SIZE)
        img = np.transpose(img, (2, 0, 1))
        
        img_list.append(img)
        
    img_list = np.array(img_list)
    img_list = np.ascontiguousarray(img_list, dtype=np.uint8)
    
    img_shape_list = np.array(img_shape_list)
    img_shape_list = np.ascontiguousarray(img_shape_list, dtype=np.float32)
        
    text_recognizer(img_list, img_shape_list)
    text_recognizer.show_text_recognition_results(args.image_dir_list)
   
   
if __name__ == "__main__":
    test_plate_detection()
    
    
    

