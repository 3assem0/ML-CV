import cv2
import random
import numpy as np

def get_random_colors(num_classes):
    """
    Generate random colors for each class.
    """
    random.seed(42)
    colors = []
    for _ in range(num_classes):
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(c)
    return colors

def draw_detections(frame, detections, colors):
    """
    Draw bounding boxes and labels on the frame.
    detections: list of dicts with 'box', 'conf', 'class_id', 'class_name'
    """
    annotated_frame = frame.copy()
    
    for det in detections:
        box = det['box']
        x1, y1, x2, y2 = map(int, box)
        conf = det['conf']
        cls_id = int(det['class_id'])
        cls_name = det['class_name']
        
        # Color for this class
        color = colors[cls_id % len(colors)]
        
        # Draw box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"{cls_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
    return annotated_frame

def count_objects(detections):
    """
    Count objects per class.
    Returns: dict {class_name: count}, total_count
    """
    counts = {}
    for det in detections:
        cls_name = det['class_name']
        counts[cls_name] = counts.get(cls_name, 0) + 1
    
    return counts, len(detections)
