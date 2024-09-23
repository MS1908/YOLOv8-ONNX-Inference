import cv2
import os
from tqdm import tqdm

from ultralytics import YOLO

detector = YOLO('model.onnx')

im_names = os.listdir('./test')
data = []
avg_infer_time = 0.
for im_name in tqdm(im_names):
    image = cv2.imread(os.path.join('./test', im_name))

    detector.predict(image, iou=0.45, conf=0.25, imgsz=640, agnostic_nms=True, save=True)
