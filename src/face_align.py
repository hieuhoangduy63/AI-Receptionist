"""
Face Alignment using MediaPipe landmarks
Aligns faces to a standard template for consistent face recognition
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple


class FaceAligner:
    # Standard face template (5 key points)
    TEMPLATE = np.array([
        [38.2946, 51.6963],  # Left eye
        [73.5318, 51.5014],  # Right eye
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Left mouth corner
        [70.7299, 92.2041]  # Right mouth corner
    ], dtype=np.float32)

    # MediaPipe landmark indices for 5 key points
    LANDMARK_INDICES = {
        "left_eye": 33,
        "right_eye": 263,
        "nose_tip": 1,
        "left_mouth": 61,
        "right_mouth": 291
    }

    def __init__(self, output_size: Tuple[int, int] = (112, 112)):
        """
        Initialize Face Aligner

        Args:
            output_size: Size of aligned face output (width, height)
        """
        self.output_size = output_size

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Scale template to output size
        self.scaled_template = self.TEMPLATE * (self.output_size[0] / 112.0)

        print("Face aligner initialized successfully")

    def get_landmarks(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 5 key facial landmarks from face crop

        Args:
            face_crop: Cropped face image

        Returns:
            5x2 array of landmark coordinates or None if detection fails
        """
        try:
            h, w = face_crop.shape[:2]

            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.face_mesh.process(face_rgb)

            if not results.multi_face_landmarks:
                return None

            # Extract landmarks
            landmarks = results.multi_face_landmarks[0].landmark

            # Get 5 key points
            key_points = []
            for point_name in ["left_eye", "right_eye", "nose_tip", "left_mouth", "right_mouth"]:
                idx = self.LANDMARK_INDICES[point_name]
                x = landmarks[idx].x * w
                y = landmarks[idx].y * h
                key_points.append([x, y])

            return np.array(key_points, dtype=np.float32)

        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None

    def align_face(self, face_crop: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Align face to standard template

        Args:
            face_crop: Cropped face image
            landmarks: Pre-computed landmarks (optional)

        Returns:
            Aligned face image or None if alignment fails
        """
        try:
            # Extract landmarks if not provided
            if landmarks is None:
                landmarks = self.get_landmarks(face_crop)

            if landmarks is None:
                return None

            # Estimate similarity transformation
            M, _ = cv2.estimateAffinePartial2D(
                landmarks,
                self.scaled_template,
                method=cv2.LMEDS
            )

            if M is None:
                return None

            # Apply transformation
            aligned_face = cv2.warpAffine(
                face_crop,
                M,
                self.output_size,
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT
            )

            return aligned_face

        except Exception as e:
            print(f"Error aligning face: {e}")
            return None

    def align_faces_from_image(self, image: np.ndarray, face_boxes: List[List[int]]) -> List[np.ndarray]:
        """
        Align multiple faces from an image given bounding boxes

        Args:
            image: Full image
            face_boxes: List of face bounding boxes [x1, y1, x2, y2]

        Returns:
            List of aligned face images
        """
        aligned_faces = []

        for box in face_boxes:
            x1, y1, x2, y2 = box

            # Add padding around face
            padding = 0.2
            w, h = x2 - x1, y2 - y1
            px, py = int(w * padding), int(h * padding)

            # Expand box with padding
            x1_pad = max(0, x1 - px)
            y1_pad = max(0, y1 - py)
            x2_pad = min(image.shape[1], x2 + px)
            y2_pad = min(image.shape[0], y2 + py)

            # Crop face
            face_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]

            if face_crop.size == 0:
                continue

            # Align face
            aligned_face = self.align_face(face_crop)

            if aligned_face is not None:
                aligned_faces.append(aligned_face)

        return aligned_faces

    def visualize_landmarks(self, face_crop: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Visualize landmarks on face crop for debugging

        Args:
            face_crop: Face image
            landmarks: 5x2 landmark coordinates

        Returns:
            Image with landmarks drawn
        """
        vis_img = face_crop.copy()

        # Draw landmarks
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis_img, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(vis_img, str(i), (int(x) + 5, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return vis_img


if __name__ == "__main__":
    # Test face alignment
    from face_detect import FaceDetector

    aligner = FaceAligner(output_size=(112, 112))
    detector = FaceDetector()

    # Test with webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Press 'q' to quit, 's' to save aligned face")
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        face_boxes, _, frame_with_boxes = detector.detect_faces(frame, draw=True)

        # Align faces
        if face_boxes:
            aligned_faces = aligner.align_faces_from_image(frame, face_boxes)

            # Show first aligned face
            if aligned_faces:
                cv2.imshow('Aligned Face', aligned_faces[0])

        cv2.imshow('Original', frame_with_boxes)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and 'aligned_faces' in locals() and aligned_faces:
            cv2.imwrite(f'aligned_face_{save_count}.jpg', aligned_faces[0])
            print(f"Saved aligned_face_{save_count}.jpg")
            save_count += 1

    cap.release()
    cv2.destroyAllWindows()