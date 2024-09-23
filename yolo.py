import cv2
import numpy as np
import onnxruntime as ort

from nms_utils import non_max_suppression
from process_utils import letterbox, scale_boxes


class YOLOv8:

    def __init__(self, path, conf_thres=0.65, iou_thres=0.2, stride=32):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        self.session = ort.InferenceSession(path,
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_size = (self.input_shape[2], self.input_shape[3])
        
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        self.stride = stride

    def preprocess(self, image):
        # Resize and pad image
        input_img, self.ratio, self.padding = letterbox(image, new_shape=self.input_size, auto=False)
        image_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        image_rgb = image_rgb.astype(np.float32) / 255.0
        
        # Transpose to match ONNX input format: (batch_size, channels, height, width)
        image_rgb = np.transpose(image_rgb, (2, 0, 1))
        
        # Add batch dimension
        preprocessed_img = np.expand_dims(image_rgb, axis=0)

        return preprocessed_img
    
    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs
    
    def postprocess(self, outputs):
        detections = non_max_suppression(outputs, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=None, agnostic=True)

        all_boxes = []
        all_scores = []
        all_class_ids = []
        for i, det in enumerate(detections):
            if len(det):
                boxes = scale_boxes(self.input_size, det[:, :4], self.img_size).round().astype(int)
                scores = det[:, 4]
                class_ids = det[:, 5].astype(int)
                
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_class_ids.append(class_ids)
            else:
                all_boxes.append(np.array([]))
                all_scores.append(np.array([]))
                all_class_ids.append(np.array([]))

        return all_boxes, all_scores, all_class_ids
    
    def __call__(self, image):
        self.img_size = (image.shape[0], image.shape[1])

        inputs = self.preprocess(image)
        outputs = self.inference(inputs)
        boxes, scores, class_ids = self.postprocess(outputs)

        # bs = 1
        boxes = boxes[0].tolist()
        scores = scores[0].tolist()
        class_ids = class_ids[0].tolist()
        return boxes, scores, class_ids
