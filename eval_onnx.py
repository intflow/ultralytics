import onnxruntime as ort
import os
import numpy as np
from tqdm import tqdm
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load ONNX model
print("Available providers:", ort.get_available_providers())

onnx_model_path = 'yolov8m_qat_opt2.onnx'
ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

# Load COCO dataset
coco_image_path = '/DL_data_super_hdd/coco_dataset/data/images/val2017'
coco_annotation_file = '/DL_data_super_hdd/coco_dataset/annotations/instances_val2017.json'
coco = COCO(coco_annotation_file)

# Create directory to save overlaid images if it doesn't exist
output_dir = "overlaid_images"
os.makedirs(output_dir, exist_ok=True)

# Define a function to draw bounding boxes on the image
def draw_bboxes(image, detections):
    for det in detections:
        x, y, w, h = map(int, det['bbox'])
        class_id = det['class_id']
        score = det['score']
        
        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Put the class ID and score on the bounding box
        label = f"{class_id}: {score:.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Define preprocess function
def preprocess(image, input_size=(640, 640)):
    image = cv2.resize(image, input_size)
    image = image.astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# Define postprocess function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess(ort_outs, input_size=(640, 640), conf_threshold=0.25, iou_threshold=0.7):
    num_classes = 80
    out0 = ort_outs[0][0]  # shape (84, 8400)
    
    # Extract bounding boxes, objectness scores, and class probabilities
    box_predictions = out0[:4, :].transpose(1, 0)  # shape (8400, 4)
    confidences = sigmoid(out0[4, :])  # shape (8400,)
    class_probs = sigmoid(out0[5:5 + num_classes, :].transpose(1, 0))  # shape (8400, 80)
    
    # Apply confidence threshold
    conf_mask = confidences > conf_threshold
    box_predictions = box_predictions[conf_mask]
    confidences = confidences[conf_mask]
    class_probs = class_probs[conf_mask]
    
    # Get scores and class IDs
    scores = confidences[:, None] * class_probs  # shape (filtered_boxes, 80)
    class_ids = np.argmax(scores, axis=1)  # shape (filtered_boxes,)
    scores = np.max(scores, axis=1)  # shape (filtered_boxes,)
    
    # Convert boxes to [x1, y1, x2, y2] format
    boxes = np.zeros_like(box_predictions)
    boxes[:, 0] = box_predictions[:, 0] - box_predictions[:, 2] / 2  # x1
    boxes[:, 1] = box_predictions[:, 1] - box_predictions[:, 3] / 2  # y1
    boxes[:, 2] = box_predictions[:, 0] + box_predictions[:, 2] / 2  # x2
    boxes[:, 3] = box_predictions[:, 1] + box_predictions[:, 3] / 2  # y2
    
    # Perform Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    indices = np.array(indices).flatten()
    
    filtered_boxes = boxes[indices]
    filtered_scores = scores[indices]
    filtered_class_ids = class_ids[indices]
    
    detections = []
    for i in range(len(filtered_boxes)):
        box = filtered_boxes[i]
        score = filtered_scores[i]
        class_id = filtered_class_ids[i]
        detections.append({
            "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # COCO format: [x, y, width, height]
            "score": float(score),
            "class_id": int(class_id)
        })
    
    return detections

# Run inference and collect results
image_ids = coco.getImgIds()
results = []

for img_id in tqdm(image_ids):
    img_info = coco.loadImgs(img_id)[0]
    image_path = coco_image_path + '/' + img_info['file_name']
    image = cv2.imread(image_path)
    input_tensor = preprocess(image)

    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Process output
    detections = postprocess(ort_outs)

    # Draw bounding boxes on the image
    image_with_bboxes = draw_bboxes(image, detections)
    
    # Save the overlaid image
    overlaid_image_path = os.path.join(output_dir, img_info['file_name'])
    cv2.imwrite(overlaid_image_path, image_with_bboxes)

    # Convert detections to COCO format and append to results
    for det in detections:
        result = {
            "image_id": img_id,
            "category_id": det['class_id'],
            "bbox": det['bbox'],  # Format: [x, y, width, height]
            "score": det['score']
        }
        results.append(result)

# Save results to file
import json
with open('results.json', 'w') as f:
    json.dump(results, f)

# Load results and run COCO evaluation
coco_dt = coco.loadRes('results.json')
coco_eval = COCOeval(coco, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
