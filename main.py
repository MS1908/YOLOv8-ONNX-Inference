import cv2
import numpy as np
from yolo import YOLOv8
from tqdm import tqdm
import os
import pandas as pd
import time


CLASS_NAMES = ['auto', 'bus', 'car', 'lcv', 'motorcycle', 'multiaxle', 'tractor', 'truck']
rng = np.random.default_rng(3)
COLORS = rng.uniform(0, 255, size=(len(CLASS_NAMES), 3))


def draw_box(image, box, color=(0, 0, 255), thickness=2):
    x1, y1, x2, y2 = box
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image, text, box, color=(0, 0, 255), font_size=0.001, text_thickness=2):
    x1, y1, x2, y2 = box
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)
    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)
    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)


def draw_masks(image, boxes, classes, mask_alpha=0.3):
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = COLORS[class_id]
        x1, y1, x2, y2 = box
        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = COLORS[class_id]
        draw_box(det_img, box, color)
        label = CLASS_NAMES[class_id]
        caption = f"{label} {int(score * 100)}%"
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def main():
    detector = YOLOv8(
        path='model.onnx',
        conf_thres=0.25,
        iou_thres=0.45
    )
    
    VISUALIZE = True
    
    im_names = os.listdir('./test')
    data = []
    avg_infer_time = 0.
    for im_name in tqdm(im_names):
        image = cv2.imread(os.path.join('./test', im_name))
        start = time.time()
        boxes, scores, class_ids = detector(image)
        avg_infer_time += time.time() - start
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            data.append([im_name, box[0], box[1], box[2], box[3], class_id, score])
    
        if VISUALIZE:
            det_img = draw_detections(image, boxes, scores, class_ids)
            os.makedirs('./visualization', exist_ok=True)
            cv2.imwrite(os.path.join('./visualization', im_name), det_img)
        
    avg_infer_time /= len(im_names)
    print(f'Average inference time: {avg_infer_time:.3f}s')
        
    df = pd.DataFrame(data, columns=['image_name', 'x0', 'y0', 'x1', 'y1', 'label', 'score'])
    df.to_csv('test_log.csv', lineterminator='\n', index=False)


if __name__ == "__main__":
    main()
