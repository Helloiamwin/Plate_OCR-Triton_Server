from ultralytics import YOLO

# Load a model
'''
# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["../PaddleOCR_clone/00207393.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
    
'''

# Export the model to ONNX format
'''
# Load a model
model = YOLO("yolo11n.pt")  # load an official model
# Export the model
model.export(format="onnx", opset = 12, dynamic = True) #, nms = True

'''

# Test the ONNX model with OpenCV
# '''
import cv2
import numpy as np
import onnxruntime as ort
import time

# === CONFIGURATION ===
MODEL_PATH = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/YOLOv11/yolo11n_dynamic.onnx'
IMAGE_PATH = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/PaddleOCR_clone/11.jpg'
IMAGE_PATH_2 = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/PaddleOCR_clone/00207393.jpg'
IMAGE_PATH_3 = '/mnt/d/Thangld/StudyMaterial/AI2product/Update_ensemble_text_det-text_rec-ocr/YOLOv11/46844597805_0c180c2ebd_b.jpg'

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.5
INPUT_SIZE = (640, 640)  # Default YOLO input size

# === PREPROCESSING ===
def preprocess(image_path):
    image = cv2.imread(image_path)
    original_shape = image.shape[:2]  # H, W
    resized = cv2.resize(image, INPUT_SIZE)
    blob = cv2.dnn.blobFromImage(resized, 1/255.0, INPUT_SIZE, swapRB=True, crop=False)
    
    return blob, image, original_shape

def preprocess_dynamic(image_path_list):
    blob_list = []
    image_list = []
    original_shape_list = []
    
    for path in image_path_list:
        if not path or not isinstance(path, str):
            raise ValueError("Each image path must be a valid string.")
        
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        
        original_shape = image.shape[:2]  # H, W
        resized = cv2.resize(image, INPUT_SIZE)
        
        # Manual preprocessing instead of blobFromImage to avoid confusion
        resized = resized[:, :, ::-1]  # BGR to RGB
        resized = resized.transpose(2, 0, 1)  # HWC to CHW
        resized = resized.astype(np.float32) / 255.0
        
        blob_list.append(resized)
        image_list.append(image)
        original_shape_list.append(original_shape)
    
    # Stack into shape: (batch, 3, H, W)
    blob = np.stack(blob_list, axis=0)
    
    return blob, image_list, original_shape_list

# === POSTPROCESSING ===
def postprocess(outputs, original_shape):
    predictions = outputs[0]  # shape (1, 84, 8400)
    predictions = np.squeeze(predictions)  # shape (84, 8400)
    predictions = predictions.T  # shape (8400, 84)

    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        # Tùy trường hợp:
        # Nếu là 84 = 4 bbox + 80 class scores:
        x, y, w, h = pred[0:4]
        class_scores = pred[4:]  # Nếu không có objectness riêng
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence > CONFIDENCE_THRESHOLD:
            # Chuyển từ cx, cy, w, h sang x, y, w, h
            x = int((x - w / 2) * original_shape[1] / INPUT_SIZE[0])
            y = int((y - h / 2) * original_shape[0] / INPUT_SIZE[1])
            width = int(w * original_shape[1] / INPUT_SIZE[0])
            height = int(h * original_shape[0] / INPUT_SIZE[1])

            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    results = []
    for i in indices:
        # i = i[0]
        results.append((boxes[i], confidences[i], class_ids[i]))
    return results

def postprocess_dynamic(outputs, original_shape):
    
    results = []
    for index, output in enumerate(outputs):
        predictions = np.squeeze(output)  # shape (84, 8400)
        predictions = predictions.T  # shape (8400, 84)

        boxes = []
        confidences = []
        class_ids = []

        for  pred in predictions:
            # Tùy trường hợp:
            x, y, w, h = pred[0:4]
            class_scores = pred[4:]  # Nếu không có objectness riêng
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                # Chuyển từ cx, cy, w, h sang x, y, w, h
                x = int((x - w / 2) * original_shape[index][1] / INPUT_SIZE[0])
                y = int((y - h / 2) * original_shape[index][0] / INPUT_SIZE[1])
                width = int(w * original_shape[index][1] / INPUT_SIZE[0])
                height = int(h * original_shape[index][0] / INPUT_SIZE[1])

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        pred_output=[(boxes[i], confidences[i], class_ids[i]) for i in indices]
            
        results.append(pred_output)

    return results

def postprocess_non_NMS(outputs, original_shape):
    predictions = np.squeeze(outputs[0])  # shape (300, 6)
    # predictions = predictions.T  # shape (8400, 84)

    results = []
    for pred in predictions:
        # Tùy trường hợp:
        x1, y1, x2, y2, class_id, confidence = pred

        if confidence > CONFIDENCE_THRESHOLD:
            x = int(x1 * original_shape[1] / INPUT_SIZE[0])
            y = int(y1  * original_shape[0] / INPUT_SIZE[1])
            width = int((x2-x1) * original_shape[1] / INPUT_SIZE[0])
            height = int((y2-y1) * original_shape[0] / INPUT_SIZE[1])
            
            results.append(([ x, y,  width, height], float(confidence), class_id))

    return results

def postprocess_non_NMS_dynamic(outputs, original_shape):
    
    results = []
    for pred in outputs:
        # predictions = outputs[0]  # shape (1, 300, 6)
        predictions = np.squeeze(pred)  # shape (300, 6)
        # predictions = predictions.T  # shape (8400, 84)

        pred_output = []
        for index, pred in enumerate (predictions):
            # Tùy trường hợp:
            x1, y1, x2, y2, class_id, confidence = pred

            if confidence > CONFIDENCE_THRESHOLD:
                x = int(x1 * original_shape[index][1] / INPUT_SIZE[0])
                y = int(y1  * original_shape[index][0] / INPUT_SIZE[1])
                width = int((x2-x1) * original_shape[index][1] / INPUT_SIZE[0])
                height = int((y2-y1) * original_shape[index][0] / INPUT_SIZE[1])
                
                pred_output.append(([ x, y,  width, height], float(confidence), class_id))
        results.append(pred_output)

    return results

# === MAIN INFERENCE ===
def run_inference(run_dynamic=False):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    input_info = session.get_inputs()[0]
    print("Input shape:", input_info.shape)

    if not run_dynamic:
        t1 = time.time()
        blob, original_image, shape = preprocess(IMAGE_PATH)
        
        t2 = time.time()
        outputs = session.run(None, {input_name: blob})
        t3 = time.time()
        
        results = postprocess_non_NMS_dynamic(outputs, shape)

        print("postprocess:", time.time() - t3, "preprocess:", t2 - t1, "inference:", t3 - t2)

        # Draw results
        for (box, conf, class_id) in results:
            x, y, w, h = box
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"ID:{class_id} {conf:.2f}"
            cv2.putText(original_image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv11 Inference", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        
        t1 = time.time()
        blob, original_image, shape = preprocess_dynamic([IMAGE_PATH_3, IMAGE_PATH, IMAGE_PATH_2])
        # Check images visually or via stats    
        t2 = time.time()
        outputs = session.run([output_name], {input_name: blob})[0]
        t3 = time.time()
        
        results = postprocess_dynamic(outputs, shape)

        print("postprocess:", time.time() - t3, "preprocess:", t2 - t1, "inference:", t3 - t2)

        image_index = 0
        # Draw results
        for (box, conf, class_id) in results[image_index]:
            x, y, w, h = box
            cv2.rectangle(original_image[image_index], (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"ID:{class_id} {conf:.2f}"
            cv2.putText(original_image[image_index], label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv11 Inference", original_image[image_index])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# === RUN ===
if __name__ == "__main__":
    run_inference(run_dynamic = True)
# '''
