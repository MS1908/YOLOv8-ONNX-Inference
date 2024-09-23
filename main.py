import cv2
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import time

from yolo import YOLOv8
from visualization import draw_detections


CLASS_NAMES = ['auto', 'bus', 'car', 'lcv', 'motorcycle', 'multiaxle', 'tractor', 'truck']
rng = np.random.default_rng(3)
COLORS = rng.uniform(0, 255, size=(len(CLASS_NAMES), 3))


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
            det_img = draw_detections(image, boxes, scores, class_ids, CLASS_NAMES, COLORS)
            os.makedirs('./visualize', exist_ok=True)
            cv2.imwrite(os.path.join('./visualize', im_name), det_img)
        
    avg_infer_time /= len(im_names)
    print(f'Average inference time: {avg_infer_time:.3f}s')
        
    df = pd.DataFrame(data, columns=['image_name', 'x0', 'y0', 'x1', 'y1', 'label', 'score'])
    df.to_csv('test_log.csv', lineterminator='\n', index=False)


if __name__ == "__main__":
    main()
