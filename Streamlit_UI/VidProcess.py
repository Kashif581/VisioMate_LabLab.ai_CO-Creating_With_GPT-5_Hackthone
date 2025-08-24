import cv2
from YOLO_Detections.Yolo_detector import VideoProcessor
from YOLO_Detections.speak_detections import DetectionLog

if __name__ == "__main__":
    detection_log = DetectionLog(repeat_interval=5.0)
    processor = VideoProcessor("yolov8n.pt", detection_log)

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = processor.process_frame(frame)

        cv2.imshow("YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
