"""
Face Encoding with Database Storage
Encodes faces using ONNX ArcFace model and stores embeddings in SQLite database
"""

import os
import sys

# Tắt hoàn toàn MediaPipe logs
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_logtostderr'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Tắt warnings
import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
import argparse
from .face_detect import FaceDetector
from .face_align import FaceAligner
from .encode_faces import OptimizedONNXFaceEncoder
from .database_manager import DatabaseManager
from typing import Dict, List, Tuple, Optional
from datetime import datetime



class FaceEncoderDB:
    def __init__(self, model_path: str = "models/arcface.onnx",
                 db_path: str = "src/face_recognition.db"):
        """
        Initialize Face Encoder with Database storage

        Args:
            model_path: Path to ArcFace ONNX model
            db_path: Path to SQLite database
        """
        self.model_path = model_path
        self.db_path = db_path

        # Initialize components
        print("Initializing Face Encoder with Database...")
        try:
            print("🤖 Loading ONNX encoder...")
            self.encoder = OptimizedONNXFaceEncoder(model_path)
            print("✅ ONNX encoder loaded")
            
            print("👁️ Loading face detector...")
            self.detector = FaceDetector()
            print("✅ Face detector loaded")
            
            print("📐 Loading face aligner...")
            self.aligner = FaceAligner(output_size=(112, 112))
            print("✅ Face aligner loaded")
            
            print("💾 Connecting to database...")
            self.db = DatabaseManager(db_path)
            print("✅ Database connected")
            
        except Exception as e:
            print(f"❌ Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
        self.encoder = OptimizedONNXFaceEncoder(model_path)
        self.detector = FaceDetector()
        self.aligner = FaceAligner(output_size=(112, 112))
        self.db = DatabaseManager(db_path)

        print("Face Encoder with Database initialized successfully!")

    def process_person_folder(self, person_folder_path: str, person_name: str,
                             person_info: Dict = None) -> Tuple[Optional[np.ndarray], int, int]:
        """
        Process all images in a person's folder and return average embedding

        Args:
            person_folder_path: Path to person's folder
            person_name: Name of the person
            person_info: Additional person information (age, major, school, etc.)

        Returns:
            Tuple of (average_embedding, number_of_faces_processed, person_id)
        """
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        all_embeddings = []
        total_images_processed = 0
        total_faces_found = 0

        print(f"\n--- Processing person: {person_name} ---")

        # Check if person already exists in database
        existing_person = self.db.get_person(name=person_name)
        person_id = None

        if existing_person:
            person_id = existing_person['id']
            print(f"  ℹ️  Person '{person_name}' already exists in database (ID: {person_id})")

            # Update person info if provided
            if person_info:
                updated = self.db.update_person(person_id=person_id, **person_info)
                if updated:
                    print(f"  ✅ Updated information for {person_name}")
        else:
            # Add new person to database
            try:
                if person_info:
                    person_id = self.db.add_person(name=person_name, **person_info)
                else:
                    person_id = self.db.add_person(name=person_name)
                print(f"  ✅ Added new person '{person_name}' to database (ID: {person_id})")
            except ValueError as e:
                print(f"  ❌ Error adding person: {e}")
                return None, 0, None

        # Get all image files in the folder
        if not os.path.exists(person_folder_path):
            print(f"  ❌ Folder not found: {person_folder_path}")
            return None, 0, person_id

        image_files = [f for f in os.listdir(person_folder_path)
                      if f.lower().endswith(supported_formats)]

        if not image_files:
            print(f"  ⚠️  No image files found in folder: {person_folder_path}")
            return None, 0, person_id

        print(f"  📁 Found {len(image_files)} images for {person_name}")

        # Process each image
        for image_file in image_files:
            try:
                image_path = os.path.join(person_folder_path, image_file)

                # Read image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"    ⚠️  Cannot read image: {image_file}")
                    continue

                total_images_processed += 1
                print(f"    📸 Processing: {image_file}")

                # Detect faces
                face_boxes, confidences, _ = self.detector.detect_faces(image)

                if not face_boxes:
                    print(f"      ❌ No faces detected")
                    continue

                # Align faces
                aligned_faces = self.aligner.align_faces_from_image(image, face_boxes)

                if not aligned_faces:
                    print(f"      ❌ No faces could be aligned")
                    continue

                # Encode faces
                valid_encodings = []
                for aligned_face in aligned_faces:
                    encoding = self.encoder.encode_face(aligned_face)
                    if encoding is not None:
                        valid_encodings.append(encoding)

                # Add valid encodings
                faces_in_image = len(valid_encodings)
                if faces_in_image > 0:
                    all_embeddings.extend(valid_encodings)
                    total_faces_found += faces_in_image
                    print(f"      ✅ Found {faces_in_image} face(s)")
                else:
                    print(f"      ❌ No valid encodings")

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

            # Save embedding to database
            try:
                embedding_id = self.db.save_face_embedding(
                    person_id=person_id,
                    embedding=average_embedding,
                    faces_processed=total_faces_found
                )
                print(f"  💾 Saved embedding to database (Embedding ID: {embedding_id})")
            except Exception as e:
                print(f"  ❌ Error saving embedding: {e}")

            return average_embedding, total_faces_found, person_id
        else:
            print(f"  ❌ No valid embeddings found for {person_name}")
            return None, 0, person_id

    def encode_dataset(self, data_folder: str = "data",
                      person_info_file: str = None) -> Dict:
        """
        Encode all faces in the dataset and store in database

        Args:
            data_folder: Path to data folder containing person subfolders
            person_info_file: Optional JSON file with person information

        Returns:
            Dictionary with processing results
        """
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        # Load person information if provided
        person_info_dict = {}
        if person_info_file and os.path.exists(person_info_file):
            try:
                import json
                with open(person_info_file, 'r', encoding='utf-8') as f:
                    person_info_dict = json.load(f)
                print(f"📄 Loaded person information from: {person_info_file}")
            except Exception as e:
                print(f"⚠️  Error loading person info file: {e}")

        print(f"🔍 Scanning data folder: {data_folder}")
        print("Expected structure: data/person_name/images...")

        # Get all person folders
        person_folders = [f for f in os.listdir(data_folder)
                         if os.path.isdir(os.path.join(data_folder, f))]

        if not person_folders:
            raise ValueError(f"No person folders found in {data_folder}")

        print(f"📁 Found {len(person_folders)} person folders")

        results = {
            'total_people': len(person_folders),
            'successfully_processed': 0,
            'failed_people': [],
            'total_faces_processed': 0,
            'processed_people': [],
            'processing_start_time': datetime.now().isoformat()
        }

        # Process each person
        for person_name in sorted(person_folders):
            person_folder_path = os.path.join(data_folder, person_name)

            # Get person info if available
            person_info = person_info_dict.get(person_name, {})

            try:
                # Process person's folder
                average_embedding, faces_count, person_id = self.process_person_folder(
                    person_folder_path, person_name, person_info
                )

                if average_embedding is not None:
                    results['successfully_processed'] += 1
                    results['total_faces_processed'] += faces_count
                    results['processed_people'].append({
                        'name': person_name,
                        'person_id': person_id,
                        'faces_processed': faces_count
                    })
                    print(f"  ✅ {person_name} successfully processed")
                else:
                    results['failed_people'].append({
                        'name': person_name,
                        'reason': 'No valid faces found'
                    })
                    print(f"  ❌ {person_name} failed (no valid faces)")

            except Exception as e:
                results['failed_people'].append({
                    'name': person_name,
                    'reason': str(e)
                })
                print(f"  ❌ {person_name} failed: {e}")

        results['processing_end_time'] = datetime.now().isoformat()

        # Print summary
        print(f"\n{'='*70}")
        print(f"🎉 Face Encoding with Database completed!")
        print(f"{'='*70}")
        print(f"👥 Total people folders: {results['total_people']}")
        print(f"✅ Successfully processed: {results['successfully_processed']}")
        print(f"❌ Failed: {len(results['failed_people'])}")
        print(f"😀 Total faces processed: {results['total_faces_processed']}")

        if results['failed_people']:
            print(f"\n❌ Failed people:")
            for failed in results['failed_people']:
                print(f"  - {failed['name']}: {failed['reason']}")

        if results['processed_people']:
            print(f"\n✅ Successfully processed people:")
            for person in results['processed_people']:
                print(f"  - {person['name']} (ID: {person['person_id']}): {person['faces_processed']} faces")

        # Show database statistics
        db_stats = self.db.get_database_stats()
        print(f"\n📊 Database Statistics:")
        print(f"  - Active people: {db_stats['active_people']}")
        print(f"  - Total embeddings: {db_stats['total_embeddings']}")
        print(f"  - Database size: {db_stats['database_size_mb']:.2f} MB")

        return results

    def encode_single_person(self, person_name: str, person_folder: str = None,
                           age: int = None, major: str = None, school: str = None,
                           meeting_room: str = None, meeting_time: str = None) -> bool:
        """
        Encode faces for a single person and add to database

        Args:
            person_name: Name of the person
            person_folder: Path to person's image folder (if None, uses ../data/person_name)
            age: Person's age
            major: Academic major
            school: School name
            meeting_room: Meeting room
            meeting_time: Meeting time

        Returns:
            True if successful
        """
        if person_folder is None:
            person_folder = os.path.join("data", person_name)

        person_info = {}
        if age is not None:
            person_info['age'] = age
        if major:
            person_info['major'] = major
        if school:
            person_info['school'] = school
        if meeting_room:
            person_info['meeting_room'] = meeting_room
        if meeting_time:
            person_info['meeting_time'] = meeting_time

        try:
            average_embedding, faces_count, person_id = self.process_person_folder(
                person_folder, person_name, person_info
            )

            if average_embedding is not None and person_id is not None:
                print(f"✅ Successfully encoded {person_name}")
                return True
            else:
                print(f"❌ Failed to encode {person_name}")
                return False

        except Exception as e:
            print(f"❌ Error encoding {person_name}: {e}")
            return False

    def update_person_embedding(self, person_name: str, person_folder: str = None) -> bool:
        """
        Update face embedding for an existing person

        Args:
            person_name: Name of the person
            person_folder: Path to person's image folder

        Returns:
            True if successful
        """
        existing_person = self.db.get_person(name=person_name)
        if not existing_person:
            print(f"❌ Person '{person_name}' not found in database")
            return False

        if person_folder is None:
            person_folder = os.path.join("data", person_name)

        try:
            print(f"🔄 Updating embedding for {person_name}...")
            average_embedding, faces_count, _ = self.process_person_folder(
                person_folder, person_name, {}
            )

            if average_embedding is not None:
                print(f"✅ Successfully updated embedding for {person_name}")
                return True
            else:
                print(f"❌ Failed to update embedding for {person_name}")
                return False

        except Exception as e:
            print(f"❌ Error updating embedding for {person_name}: {e}")
            return False

    def list_people_without_embeddings(self) -> List[str]:
        """
        List people in database who don't have face embeddings

        Returns:
            List of person names without embeddings
        """
        try:
            # Get all people from database
            all_people = self.db.list_all_people()

            # Get people with embeddings
            _, names_with_embeddings, _ = self.db.get_all_embeddings()

            # Find people without embeddings
            people_without_embeddings = []
            for person in all_people:
                if person['name'] not in names_with_embeddings:
                    people_without_embeddings.append(person['name'])

            return people_without_embeddings

        except Exception as e:
            print(f"Error finding people without embeddings: {e}")
            return []


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Face Encoding with Database Storage')
    parser.add_argument('--model', type=str, default='models/arcface.onnx',
                       help='Path to ArcFace ONNX model')
    parser.add_argument('--database', type=str, default='src/face_recognition.db',
                       help='Path to SQLite database')
    parser.add_argument('--data', type=str, default='src/data',
                       help='Path to data folder')
    parser.add_argument('--person-info', type=str,
                       help='Path to JSON file with person information')
    parser.add_argument('--single-person', type=str,
                       help='Encode only specific person')
    parser.add_argument('--person-folder', type=str,
                       help='Folder path for single person (used with --single-person)')
    parser.add_argument('--update-embedding', type=str,
                       help='Update embedding for specific person')
    parser.add_argument('--list-no-embeddings', action='store_true',
                       help='List people without face embeddings')

    args = parser.parse_args()

    print("=" * 70)
    print("Face Encoding with Database Storage")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Database: {args.database}")
    print(f"Data folder: {args.data}")
    if args.person_info:
        print(f"Person info file: {args.person_info}")
    print("=" * 70)

    try:
        # Initialize encoder
        encoder = FaceEncoderDB(
            model_path=args.model,
            db_path=args.database
        )

        if args.single_person:
            # Encode single person
            print(f"Encoding single person: {args.single_person}")
            success = encoder.encode_single_person(
                person_name=args.single_person,
                person_folder=args.person_folder
            )
            if success:
                print("✅ Single person encoding completed successfully!")
            else:
                print("❌ Single person encoding failed!")

        elif args.update_embedding:
            # Update embedding for existing person
            print(f"Updating embedding for: {args.update_embedding}")
            success = encoder.update_person_embedding(
                person_name=args.update_embedding,
                person_folder=args.person_folder
            )
            if success:
                print("✅ Embedding update completed successfully!")
            else:
                print("❌ Embedding update failed!")

        elif args.list_no_embeddings:
            # List people without embeddings
            people_without_embeddings = encoder.list_people_without_embeddings()
            if people_without_embeddings:
                print("👥 People without face embeddings:")
                for person_name in people_without_embeddings:
                    print(f"  - {person_name}")
            else:
                print("✅ All people in database have face embeddings!")

        else:
            # Encode entire dataset
            results = encoder.encode_dataset(
                data_folder=args.data,
                person_info_file=args.person_info
            )

            print(f"\n🎯 Final Results:")
            print(f"  Processed: {results['successfully_processed']}/{results['total_people']}")
            print(f"  Total faces: {results['total_faces_processed']}")

            if results['successfully_processed'] > 0:
                print("✅ Dataset encoding completed successfully!")
            else:
                print("❌ No people were successfully processed!")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the data folder exists with person subfolders")
        print("2. Check that the ONNX model file exists")
        print("3. Verify folder structure: data/person_name/images...")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check that all required files exist")
        print("2. Verify folder permissions")
        print("3. Install required packages: pip install opencv-python onnxruntime numpy")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()