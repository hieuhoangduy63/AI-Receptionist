"""
Optimized Face Encoding using ONNX ArcFace model with Average Embeddings per Person
Based on model specs: Input NHWC (112,112,3), Output (512,) embedding
"""
import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # tránh lỗi SIMD/AVX

import cv2
import numpy as np
import pickle
import onnxruntime as ort
try:
    from .face_detect import FaceDetector
    from .face_align import FaceAligner
except ImportError:
    from face_detect import FaceDetector
    from face_align import FaceAligner
from typing import Dict, List, Tuple
import json
from datetime import datetime


class OptimizedONNXFaceEncoder:
    def __init__(self, model_path: str = "models/arcface.onnx"):
        """
        Initialize Face Encoder with ONNX ArcFace model

        Model specs from debug:
        - Input: ['unk__556', 112, 112, 3] NHWC format
        - Output: ['unk__557', 512] - 512-dim embedding
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.load_model()

        # Initialize detector and aligner
        self.detector = FaceDetector()
        self.aligner = FaceAligner(output_size=(112, 112))
        

        print("Optimized ONNX Face encoder initialized successfully")

    def load_model(self):
        """Load ONNX ArcFace model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Create ONNX Runtime session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )

            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name  # Should be 'input_1'
            self.output_name = self.session.get_outputs()[0].name  # Should be 'embedding'

            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape

            print(f"ONNX ArcFace model loaded from {self.model_path}")
            print(f"Input: {self.input_name} {input_shape}")
            print(f"Output: {self.output_name} {output_shape}")
            print(f"Providers: {self.session.get_providers()}")

            # Verify with test input
            self.verify_model()

        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise

    def verify_model(self):
        """Verify model works with test input"""
        try:
            print("Verifying model...")
            test_input = np.random.rand(1, 112, 112, 3).astype(np.float32)
            result = self.session.run([self.output_name], {self.input_name: test_input})

            embedding = result[0]
            print(f"✅ Model verification successful")
            print(f"   Test embedding shape: {embedding.shape}")
            print(f"   Test embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")

            # Check if already normalized
            l2_norm = np.linalg.norm(embedding)
            print(f"   L2 norm: {l2_norm:.4f}")
            if abs(l2_norm - 1.0) < 0.01:
                print("   ✅ Output is L2 normalized")
                self.needs_normalization = False
            else:
                print("   ⚠️  Output needs L2 normalization")
                self.needs_normalization = True

        except Exception as e:
            print(f"Model verification failed: {e}")
            raise

    def preprocess_face(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face for ONNX ArcFace model

        Model expects: NHWC format, shape (1, 112, 112, 3), float32
        """
        # Ensure correct size
        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112))

        # Convert BGR to RGB
        if len(aligned_face.shape) == 3 and aligned_face.shape[2] == 3:
            face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = aligned_face

        # Normalize to [0, 1] - most common for ArcFace
        face_normalized = face_rgb.astype(np.float32) / 255.0

        # Model expects NHWC format (batch, height, width, channels)
        # Add batch dimension: HWC -> NHWC
        return np.expand_dims(face_normalized, axis=0)

    def encode_face(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Encode aligned face to 512-dim feature vector

        Args:
            aligned_face: Aligned face image (112x112)

        Returns:
            512-dimensional face embedding
        """
        try:
            # Preprocess face
            face_input = self.preprocess_face(aligned_face)

            # Run inference
            result = self.session.run([self.output_name], {self.input_name: face_input})
            embedding = result[0]  # Shape: (1, 512)

            # Apply L2 normalization if needed
            if self.needs_normalization:
                norm = np.linalg.norm(embedding, axis=1, keepdims=True)
                norm = np.where(norm == 0, 1, norm)  # Avoid division by zero
                embedding = embedding / norm

            # Return flattened embedding
            return embedding.flatten()  # Shape: (512,)

        except Exception as e:
            print(f"Error encoding face: {e}")
            import traceback
            traceback.print_exc()
            return None

    def encode_faces_batch(self, aligned_faces: List[np.ndarray]) -> List[np.ndarray]:
        """
        Encode multiple faces in batch for better performance

        Args:
            aligned_faces: List of aligned face images

        Returns:
            List of face embeddings
        """
        try:
            if not aligned_faces:
                return []

            # Preprocess all faces
            batch_input = []
            for face in aligned_faces:
                preprocessed = self.preprocess_face(face)
                batch_input.append(preprocessed[0])  # Remove batch dim for concatenation

            # Stack into batch
            batch_input = np.stack(batch_input, axis=0)  # Shape: (N, 112, 112, 3)

            # Run batch inference
            result = self.session.run([self.output_name], {self.input_name: batch_input})
            embeddings = result[0]  # Shape: (N, 512)

            # Apply L2 normalization if needed
            if self.needs_normalization:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                embeddings = embeddings / norms

            # Return list of individual embeddings
            return [emb for emb in embeddings]

        except Exception as e:
            print(f"Error in batch encoding: {e}")
            # Fallback to individual encoding
            return [self.encode_face(face) for face in aligned_faces]

    def process_person_folder(self, person_folder_path: str, person_name: str) -> Tuple[np.ndarray, int]:
        """
        Process all images in a person's folder and return average embedding

        Args:
            person_folder_path: Path to person's folder
            person_name: Name of the person

        Returns:
            Tuple of (average_embedding, number_of_faces_processed)
        """
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        all_embeddings = []
        total_images_processed = 0
        total_faces_found = 0

        print(f"\n--- Processing person: {person_name} ---")

        # Get all image files in the folder
        image_files = [f for f in os.listdir(person_folder_path)
                      if f.lower().endswith(supported_formats)]

        if not image_files:
            print(f"No image files found in folder: {person_folder_path}")
            return None, 0

        print(f"Found {len(image_files)} images for {person_name}")

        for image_file in image_files:
            try:
                image_path = os.path.join(person_folder_path, image_file)

                # Read image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  ⚠️  Cannot read image: {image_file}")
                    continue

                total_images_processed += 1
                print(f"  📸 Processing: {image_file}")

                # Detect faces
                face_boxes, confidences, _ = self.detector.detect_faces(image)

                if not face_boxes:
                    print(f"    ❌ No faces detected")
                    continue

                # Align faces
                aligned_faces = self.aligner.align_faces_from_image(image, face_boxes)

                if not aligned_faces:
                    print(f"    ❌ No faces could be aligned")
                    continue

                # Encode faces (use batch processing for multiple faces)
                if len(aligned_faces) > 1:
                    encodings = self.encode_faces_batch(aligned_faces)
                    valid_encodings = [enc for enc in encodings if enc is not None]
                else:
                    encoding = self.encode_face(aligned_faces[0])
                    valid_encodings = [encoding] if encoding is not None else []

                # Add valid encodings
                faces_in_image = len(valid_encodings)
                if faces_in_image > 0:
                    all_embeddings.extend(valid_encodings)
                    total_faces_found += faces_in_image
                    print(f"    ✅ Found {faces_in_image} face(s)")
                else:
                    print(f"    ❌ No valid encodings")

            except Exception as e:
                print(f"    ❌ Error processing {image_file}: {e}")
                continue

        # Calculate average embedding if we have any embeddings
        if all_embeddings:
            # Convert to numpy array and calculate mean
            embeddings_array = np.array(all_embeddings)
            average_embedding = np.mean(embeddings_array, axis=0)

            # Normalize the average embedding
            norm = np.linalg.norm(average_embedding)
            if norm > 0:
                average_embedding = average_embedding / norm

            print(f"  🎯 Final result: {total_faces_found} faces from {total_images_processed} images")
            print(f"  📊 Average embedding shape: {average_embedding.shape}")
            print(f"  📈 Embedding norm: {np.linalg.norm(average_embedding):.4f}")

            return average_embedding, total_faces_found
        else:
            print(f"  ❌ No valid embeddings found for {person_name}")
            return None, 0

    def encode_dataset(self, data_folder: str = "src/data", output_file: str = "face_encodings_averaged.pkl") -> Dict:
        """
        Encode all faces in the dataset and create average embeddings per person
        Expected folder structure:
        data/
        ├── person1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── person2/
        │   ├── image1.jpg
        │   └── ...
        """
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        encodings_db = {
            'encodings': [],
            'names': [],
            'person_stats': {},  # Store statistics for each person
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_people': 0,
                'total_faces_processed': 0,
                'model_path': self.model_path,
                'model_type': 'ONNX_ArcFace_Averaged',
                'embedding_dim': 512,
                'input_format': 'NHWC',
                'normalized': not self.needs_normalization,
                'encoding_method': 'average_per_person'
            }
        }

        print(f"Scanning data folder: {data_folder}")
        print("Expected structure: data/person_name/images...")

        # Get all person folders
        person_folders = [f for f in os.listdir(data_folder)
                         if os.path.isdir(os.path.join(data_folder, f))]

        if not person_folders:
            raise ValueError(f"No person folders found in {data_folder}")

        print(f"Found {len(person_folders)} person folders")

        successfully_processed = 0
        total_faces_processed = 0

        for person_name in person_folders:
            person_folder_path = os.path.join(data_folder, person_name)

            # Process person's folder
            average_embedding, faces_count = self.process_person_folder(
                person_folder_path, person_name
            )

            if average_embedding is not None:
                # Add to database
                encodings_db['encodings'].append(average_embedding)
                encodings_db['names'].append(person_name)

                # Store person statistics
                encodings_db['person_stats'][person_name] = {
                    'faces_processed': faces_count,
                    'embedding_norm': float(np.linalg.norm(average_embedding))
                }

                successfully_processed += 1
                total_faces_processed += faces_count

                print(f"  ✅ {person_name} added to database")
            else:
                print(f"  ❌ {person_name} skipped (no valid faces)")

        # Update metadata
        encodings_db['metadata']['total_people'] = successfully_processed
        encodings_db['metadata']['total_faces_processed'] = total_faces_processed
        encodings_db['metadata']['people_list'] = list(encodings_db['names'])

        # Convert encodings to numpy array
        if encodings_db['encodings']:
            encodings_db['encodings'] = np.array(encodings_db['encodings'])

        # Save results
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(encodings_db, f)

            print(f"\n🎉 ONNX ArcFace Encoding completed!")
            print(f"📁 Saved to: {output_file}")
            print(f"👥 Successfully processed people: {successfully_processed}")
            print(f"😀 Total faces processed: {total_faces_processed}")
            print(f"🧠 Embedding dimension: 512 (averaged per person)")
            print(f"📋 People in database: {', '.join(encodings_db['names'])}")

            # Save metadata
            metadata_file = output_file.replace('.pkl', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(encodings_db['metadata'], f, indent=2)
            print(f"📄 Metadata saved to: {metadata_file}")

            # Save detailed statistics
            stats_file = output_file.replace('.pkl', '_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(encodings_db['person_stats'], f, indent=2)
            print(f"📊 Statistics saved to: {stats_file}")

        except Exception as e:
            print(f"Error saving encodings: {e}")
            raise

        return encodings_db

    def load_encodings(self, encodings_file: str = "face_encodings_averaged.pkl") -> Dict:
        """Load averaged encodings from file"""
        try:
            with open(encodings_file, 'rb') as f:
                encodings_db = pickle.load(f)

            print(f"Loaded averaged ONNX encodings:")
            print(f"  👥 People: {len(encodings_db['names'])}")
            print(f"  😀 Total faces processed: {encodings_db['metadata']['total_faces_processed']}")
            print(f"  🧠 Encoding method: {encodings_db['metadata']['encoding_method']}")

            return encodings_db

        except Exception as e:
            print(f"Error loading encodings: {e}")
            raise


if __name__ == "__main__":
    try:
        # Create optimized ONNX face encoder
        encoder = OptimizedONNXFaceEncoder(model_path="models/arcface.onnx")

        # Encode dataset with average embeddings per person
        print("Starting optimized ONNX face encoding with averaging...")
        print("Expected folder structure:")
        print("data/")
        print("├── person1/")
        print("│   ├── image1.jpg")
        print("│   ├── image2.jpg")
        print("│   └── ...")
        print("├── person2/")
        print("│   └── ...")
        print()

        encodings_db = encoder.encode_dataset(
            data_folder="src/data",
            output_file="face_encodings_averaged.pkl"
        )

        print("\n" + "=" * 70)
        print("🎉 Optimized ONNX Face encoding with averaging completed!")
        print("=" * 70)

        # Test loading
        print("\nTesting loading...")
        loaded_db = encoder.load_encodings("face_encodings_averaged.pkl")

        # Show some statistics
        if loaded_db['person_stats']:
            print("\nPer-person statistics:")
            for person, stats in loaded_db['person_stats'].items():
                print(f"  {person}: {stats['faces_processed']} faces processed")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. Data folder exists at 'src/data' with person subfolders")
        print("2. Each person subfolder contains images")
        print("3. ONNX model exists at 'models/arcface.onnx'")
        print("4. Required packages: pip install onnxruntime opencv-python")
        import traceback
        traceback.print_exc()