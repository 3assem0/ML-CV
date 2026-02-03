import cv2
import argparse
import sys
from detector import ObjectDetector
from utils import draw_detections, get_random_colors, count_objects

def main():
    parser = argparse.ArgumentParser(description="Object Detection on Image")
    parser.add_argument("--image", type=str, help="Path to input image", required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    # Initialize Detector
    detector = ObjectDetector(model_path=args.model)
    colors = get_random_colors(len(detector.class_names))

    # Load Image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not open or find the image {args.image}")
        sys.exit(1)

    # Detect
    detections, _ = detector.detect(image, conf_threshold=args.conf)
    
    # Process Results
    annotated_image = draw_detections(image, detections, colors)
    counts, total = count_objects(detections)

    print(f"Total objects detected: {total}")
    for cls, count in counts.items():
        print(f" - {cls}: {count}")

    # Show Output
    cv2.imshow("Detected Image", annotated_image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
