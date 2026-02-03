from ultralytics import YOLO
import logging
import os

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize YOLOv8 model.
        It will automatically download the model if it doesn't exist locally
        or in the default Ultralytics cache.
        """
        self.model_path = model_path
        logging.info(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)  # This loads the model
        self.class_names = self.model.names

    def detect(self, frame, conf_threshold=0.5):
        """
        Run detection on a single frame (numpy array).
        Returns: list of detections
        """
        results = self.model.predict(source=frame, conf=conf_threshold, verbose=False)
        result = results[0]  # We only process one frame
        
        detections = []
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = self.class_names[cls_id]
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': conf,
                'class_id': cls_id,
                'class_name': cls_name
            })
            
        return detections, self.class_names
