"""
Face Detector using YOLOv8
Detects faces in images and returns bounding boxes
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional


class FaceDetector:
    def __init__(self, model_path: str = '..\models\yolov8x-face-lindevs.pt', device: Optional[str] = None, conf_thres: float = 0.5):
        """
        Initialize Face Detector with YOLOv8

        Args:
            model_path: Path to YOLO face detection model
            device: Device to run inference on ('cpu', 'cuda', etc.)
            conf_thres: Confidence threshold for face detection
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_thres = conf_thres

        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"Face detector loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading face detector: {e}")
            raise

    def detect_faces(self, image: np.ndarray, draw: bool = False) -> Tuple[List[List[int]], List[float], np.ndarray]:
        """
        Detect faces in an image

        Args:
            image: Input image as numpy array
            draw: Whether to draw bounding boxes on the image

        Returns:
            Tuple of (bounding_boxes, confidences, processed_image)
        """
        try:
            # Run YOLO inference
            results = self.model.predict(image, verbose=False, conf=self.conf_thres)

            bboxes = []
            confidences = []

            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    # Extract coordinates and confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])

                    # Ensure coordinates are within image bounds
                    h, w = image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # Only add if valid box
                    if x2 > x1 and y2 > y1:
                        bboxes.append([x1, y1, x2, y2])
                        confidences.append(conf)

                        if draw:
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(image, f'{conf:.2f}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            return bboxes, confidences, image

        except Exception as e:
            print(f"Error in face detection: {e}")
            return [], [], image

    def get_largest_face(self, image: np.ndarray) -> Optional[List[int]]:
        """
        Get the largest face bounding box from an image

        Args:
            image: Input image

        Returns:
            Bounding box of largest face or None if no face found
        """
        bboxes, confidences, _ = self.detect_faces(image)

        if not bboxes:
            return None

        # Find largest face by area
        max_area = 0
        largest_box = None

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_box = bbox

        return largest_box


if __name__ == "__main__":
    # Test the face detector
    detector = FaceDetector(conf_thres=0.5)

    # Test with webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces and draw
        _, _, frame_with_faces = detector.detect_faces(frame, draw=True)

        cv2.imshow('Face Detection Test', frame_with_faces)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()