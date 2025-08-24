import cv2
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, model_path="YOLO_models\yolov8n.pt", detection_log=None):
        self.model = YOLO(model_path)
        self.detection_log = detection_log

        # Only detect these classes
        self.allowed_classes = ["person", "chair", "dining table", "bed"]

    def process_frame(self, frame):
        results = self.model(frame)

        detections = []
        h, w, _ = frame.shape
        left_bound = w // 3
        right_bound = 2 * w // 3

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]

            # Skip if not in target classes
            if label not in self.allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            # Find center X of object
            center_x = (x1 + x2) // 2

            # Determine direction
            if center_x < left_bound:
                direction = f"{label} left move right"
            elif center_x > right_bound:
                direction = f"{label} right move left"
            else:
                direction = f"{label} center move left or right"

            detections.append(direction)

        # Log and speak detections if available
        if self.detection_log and detections:
            for d in detections:
                self.detection_log.log(d)

        return frame
