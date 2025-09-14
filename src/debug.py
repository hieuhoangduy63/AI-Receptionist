import os
import cv2
from .face_detect import FaceDetector
from .face_align import FaceAligner

def test_align_image(image_path: str, output_dir: str = "aligned_preview"):
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot read image: {image_path}")
        return

    print(f"\n📸 Testing align on: {os.path.basename(image_path)}")

    # Phát hiện khuôn mặt
    detector = FaceDetector()
    face_boxes, confidences, frame_with_boxes = detector.detect_faces(image, draw=True)

    if not face_boxes:
        print("❌ No faces detected")
        return

    print(f"✅ Detected {len(face_boxes)} face(s)")

    # Căn chỉnh khuôn mặt
    aligner = FaceAligner(output_size=(112, 112))
    aligned_faces = aligner.align_faces_from_image(image, face_boxes)

    if not aligned_faces:
        print("❌ No faces could be aligned")
        return

    # Hiển thị và lưu kết quả
    for i, aligned_face in enumerate(aligned_faces):
        out_path = os.path.join(output_dir, f"aligned_{i}.jpg")
        cv2.imwrite(out_path, aligned_face)
        print(f"✅ Saved aligned face {i+1} to {out_path}")
        cv2.imshow(f"Aligned Face {i+1}", aligned_face)

    cv2.imshow("Original with Boxes", frame_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Thay đường dẫn ảnh cần test ở đây
    test_align_image(r"src/data/Hieu/Hieu1.jpg")
