"""
Real-time Face Recognition using Database Storage with Session Man        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.recognition_stats = {}
        self._last_recognized_person = ""  # Track last recognized person for chat

        print("Face recognition system with database and session management initialized successfully!")
Uses ArcFace ONNX model for face recognition with database-stored embeddings
Enhanced with smart session management for stable recognition
"""

import cv2
import numpy as np
import time
from datetime import datetime
from .face_detect import FaceDetector
from .face_align import FaceAligner
from .encode_faces import OptimizedONNXFaceEncoder as FaceEncoder
from .database_manager import DatabaseManager
from .face_session_manager import FaceSessionManager
from .enhanced_chatbot import EnhancedChatBot, TTSHandler
from typing import Dict, List, Tuple, Optional
import argparse


class FaceRecognitionSystemDB:
    def __init__(self, db_path: str = "face_recognition.db",
                 model_path: str = "../models/arcface.onnx",
                 similarity_threshold: float = 0.6,
                 min_confirm_frames: int = 5,
                 cooldown_seconds: int = 300,
                 lost_reset_seconds: float = 2.0):
        """
        Initialize Face Recognition System with Database and Session Management

        Args:
            db_path: Path to SQLite database
            model_path: Path to ArcFace ONNX model
            similarity_threshold: Threshold for face recognition (0.4-0.8 recommended)
            min_confirm_frames: Minimum consecutive frames to confirm recognition
            cooldown_seconds: Cooldown time before greeting same person again (seconds)
            lost_reset_seconds: Time to wait before resetting session when person is lost
        """
        self.similarity_threshold = similarity_threshold
        self.db_path = db_path
        self.model_path = model_path

        # Initialize components
        print("Initializing face recognition system with database and session management...")
        self.detector = FaceDetector()
        self.aligner = FaceAligner(output_size=(112, 112))
        self.encoder = FaceEncoder(model_path)
        self.db = DatabaseManager(db_path)
        
        # Initialize session manager
        self.session_manager = FaceSessionManager(
            min_confirm_frames=min_confirm_frames,
            cooldown_seconds=cooldown_seconds,
            lost_reset_seconds=lost_reset_seconds
        )
        
        # Initialize chatbot and TTS
        try:
            self.chatbot = EnhancedChatBot()
            self.tts_handler = TTSHandler()
            print("✅ Chatbot and TTS initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize chatbot/TTS: {e}")
            self.chatbot = None
            self.tts_handler = None

        # Load known face encodings from database
        self.known_encodings = None
        self.known_names = None
        self.person_info_list = None
        self.load_known_faces()

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.recognition_stats = {}

        print("Face recognition system with database initialized successfully!")

    def load_known_faces(self):
        """Load known face encodings from database"""
        try:
            embeddings, names, person_info = self.db.get_all_embeddings()

            if embeddings and len(embeddings) > 0:  # Fix: Check if embeddings exist and not empty
                self.known_encodings = np.array(embeddings)
                self.known_names = names
                self.person_info_list = person_info

                unique_people = len(set(names))
                total_faces = sum(info.get('faces_processed', 0) for info in person_info)  # Fix: Use .get() with default

                print(f"Loaded {len(embeddings)} face embeddings from database")
                print(f"Known people: {unique_people}")
                print(f"Total faces processed: {total_faces}")

                # Display per-person statistics
                person_stats = {}
                for name, info in zip(names, person_info):
                    if name not in person_stats:
                        person_stats[name] = info.get('faces_processed', 0)  # Fix: Use .get() with default

                print("\nPer-person statistics:")
                for person, faces_count in person_stats.items():
                    print(f"  {person}: {faces_count} faces")

            else:
                print("No face encodings found in database")
                self.known_encodings = None
                self.known_names = None
                self.person_info_list = None

        except Exception as e:
            print(f"Error loading known faces from database: {e}")
            self.known_encodings = None
            self.known_names = None
            self.person_info_list = None

    def calculate_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two face encodings

        Args:
            encoding1: First face encoding
            encoding2: Second face encoding

        Returns:
            Similarity score (0-1, higher means more similar)
        """
        try:
            # Ensure encodings are numpy arrays
            if not isinstance(encoding1, np.ndarray):
                encoding1 = np.array(encoding1)
            if not isinstance(encoding2, np.ndarray):
                encoding2 = np.array(encoding2)

            # Check if encodings have the right shape
            if encoding1.ndim != 1 or encoding2.ndim != 1:
                print(f"Warning: Unexpected encoding shapes: {encoding1.shape}, {encoding2.shape}")
                return 0.0

            # Ensure encodings are normalized
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            encoding1 = encoding1 / norm1
            encoding2 = encoding2 / norm2

            # Cosine similarity
            similarity = np.dot(encoding1, encoding2)

            # Clamp to [0, 1] range
            similarity = np.clip(similarity, 0, 1)

            return float(similarity)

        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def recognize_face(self, face_encoding: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Recognize face from encoding

        Args:
            face_encoding: Face encoding to recognize

        Returns:
            Tuple of (name, confidence_score, person_info)
        """
        # Fix: More robust checking
        if (self.known_encodings is None or self.known_names is None or
                len(self.known_encodings) == 0 or len(self.known_names) == 0):
            print("DEBUG: No known encodings loaded!")
            return "Unknown", 0.0, {}

        print(f"DEBUG: Comparing with {len(self.known_encodings)} known faces")
        print(f"DEBUG: Face encoding norm: {np.linalg.norm(face_encoding):.6f}")

        try:
            # Calculate similarities with all known faces
            similarities = []
            for i, known_encoding in enumerate(self.known_encodings):
                similarity = self.calculate_similarity(face_encoding, known_encoding)
                similarities.append(similarity)
                print(f"DEBUG: Similarity with {self.known_names[i]}: {similarity:.6f}")

            if not similarities:  # Fix: Check if similarities list is empty
                print("DEBUG: No similarities calculated!")
                return "Unknown", 0.0, {}

            similarities = np.array(similarities)

            # Find best match
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]

            print(f"DEBUG: Best match: {self.known_names[best_match_idx]} with similarity {best_similarity:.6f}")
            print(f"DEBUG: Threshold: {self.similarity_threshold}")

            if best_similarity >= self.similarity_threshold:
                name = self.known_names[best_match_idx]
                person_info = self.person_info_list[best_match_idx] if self.person_info_list else {}

                # Update recognition stats
                if name not in self.recognition_stats:
                    self.recognition_stats[name] = {'count': 0, 'last_seen': time.time()}
                self.recognition_stats[name]['count'] += 1
                self.recognition_stats[name]['last_seen'] = time.time()

                print(f"DEBUG: RECOGNIZED as {name}!")
                return name, best_similarity, person_info
            else:
                print(f"DEBUG: Below threshold ({best_similarity:.6f} < {self.similarity_threshold})")
                return "Unknown", best_similarity, {}

        except Exception as e:
            print(f"Error in recognize_face: {e}")
            import traceback
            traceback.print_exc()
            return "Error", 0.0, {}

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        elapsed_time = time.time() - self.fps_start_time

        if elapsed_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = time.time()

    def format_person_info(self, person_info: Dict) -> List[str]:
        """Format person information for display"""
        info_lines = []

        if person_info.get('age'):
            info_lines.append(f"Age: {person_info['age']}")
        if person_info.get('major'):
            info_lines.append(f"Major: {person_info['major']}")
        if person_info.get('school'):
            info_lines.append(f"School: {person_info['school']}")
        if person_info.get('meeting_room'):
            info_lines.append(f"Room: {person_info['meeting_room']}")
        if person_info.get('meeting_time'):
            info_lines.append(f"Meeting: {person_info['meeting_time']}")

        return info_lines

    def draw_results(self, frame: np.ndarray, face_boxes: List[List[int]],
                     names: List[str], confidences: List[float],
                     person_infos: List[Dict]) -> np.ndarray:
        """
        Draw recognition results on frame

        Args:
            frame: Input frame
            face_boxes: List of face bounding boxes
            names: List of recognized names
            confidences: List of confidence scores
            person_infos: List of person information dictionaries

        Returns:
            Frame with results drawn
        """
        result_frame = frame.copy()

        for i, (box, name, confidence, person_info) in enumerate(zip(face_boxes, names, confidences, person_infos)):
            x1, y1, x2, y2 = box

            # Choose color based on recognition
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown ({confidence:.3f})"
            else:
                color = (0, 255, 0)  # Green for known
                faces_count = person_info.get('faces_processed', 0)
                label = f"{name} ({confidence:.3f})"
                if faces_count > 0:
                    label += f" [{faces_count}f]"

            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            # Calculate label size and position
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10

            # Draw label background
            cv2.rectangle(result_frame, (x1, label_y - label_size[1] - 5),
                          (x1 + label_size[0], label_y + 5), color, -1)

            # Draw label text
            cv2.putText(result_frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw additional person info for recognized faces
            if name != "Unknown" and person_info:
                info_lines = self.format_person_info(person_info)
                if info_lines:
                    info_y = y2 + 20
                    for info_line in info_lines:
                        info_size = cv2.getTextSize(info_line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        # Draw info background
                        cv2.rectangle(result_frame, (x1, info_y - info_size[1] - 2),
                                      (x1 + info_size[0], info_y + 2), (0, 0, 0), -1)
                        # Draw info text
                        cv2.putText(result_frame, info_line, (x1, info_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        info_y += 18

        return result_frame

    def draw_info_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw info panel on frame"""
        h, w = frame.shape[:2]

        # Draw semi-transparent background for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Get database stats
        try:
            db_stats = self.db.get_database_stats()
        except:
            db_stats = {}

        unique_people = len(set(self.known_names)) if self.known_names else 0
        total_embeddings = len(self.known_encodings) if self.known_encodings is not None else 0

        # Draw info text
        info_text = [
            f"FPS: {self.current_fps:.1f}",
            f"Known people: {unique_people}",
            f"Total embeddings: {total_embeddings}",
            f"Active people in DB: {db_stats.get('active_people', 0)}",
            f"Threshold: {self.similarity_threshold:.3f}",
            f"Model: ONNX ArcFace (Database)",
            f"DB Size: {db_stats.get('database_size_mb', 0):.2f} MB",
            f"Press 'q' to quit, 'r' to reload from DB",
            f"Press '+'/'-' to adjust threshold",
            f"Press 's' to save frame, 'i' to toggle info"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (20, 30 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def draw_recognition_stats(self, frame: np.ndarray) -> np.ndarray:
        """Draw recognition statistics on frame"""
        if not self.recognition_stats:
            return frame

        h, w = frame.shape[:2]

        # Position stats panel on the right side
        panel_x = w - 250
        panel_y = 10
        panel_height = min(200, 20 + len(self.recognition_stats) * 20)

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, panel_y + panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Draw title
        cv2.putText(frame, "Recognition Stats:", (panel_x + 10, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw stats for each person
        y_offset = 40
        for name, stats in sorted(self.recognition_stats.items()):
            time_since = time.time() - stats['last_seen']
            if time_since < 60:  # Only show recent recognitions
                text = f"{name}: {stats['count']}"
                cv2.putText(frame, text, (panel_x + 10, panel_y + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20

        return frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str], List[float], List[Dict]]:
        """
        Process single frame for face recognition with session management

        Args:
            frame: Input frame

        Returns:
            Tuple of (processed_frame, names, confidences, person_infos)
        """
        try:
            # Detect faces
            face_boxes, _, _ = self.detector.detect_faces(frame)

            names = []
            confidences = []
            person_infos = []
            session_actions = []

            # Fix: Check if face_boxes is not None and not empty
            if face_boxes is not None and len(face_boxes) > 0:
                # Align faces
                aligned_faces = self.aligner.align_faces_from_image(frame, face_boxes)

                # Recognize each face
                for aligned_face in aligned_faces:
                    if aligned_face is not None:
                        try:
                            # Encode face using ONNX model
                            encoding = self.encoder.encode_face(aligned_face)

                            # Fix: More robust checking of encoding
                            if (encoding is not None and
                                isinstance(encoding, (np.ndarray, list)) and
                                len(encoding) > 0):
                                # Convert to numpy array if it's a list
                                if isinstance(encoding, list):
                                    encoding = np.array(encoding)

                                # Recognize face
                                name, confidence, person_info = self.recognize_face(encoding)
                                
                                # Process with session manager
                                if name != "Unknown" and confidence > self.similarity_threshold:
                                    # Get person ID (use name as ID for now)
                                    person_id = person_info.get('id', name)
                                    
                                    # Track last recognized person for chat
                                    self._last_recognized_person = name
                                    
                                    # Process detection with session manager
                                    session_result = self.session_manager.process_detection(
                                        person_id=person_id,
                                        person_name=name,
                                        confidence=confidence
                                    )
                                    
                                    session_actions.append(session_result)
                                    
                                    # Handle auto-greeting if needed (có cooldown)
                                    if session_result['should_greet']:
                                        self._handle_greeting(name, person_info, session_result)
                                
                                names.append(name)
                                confidences.append(confidence)
                                person_infos.append(person_info)
                                
                            else:
                                names.append("Encode Error")
                                confidences.append(0.0)
                                person_infos.append({})
                                session_actions.append({'action': 'encode_error'})
                                
                        except Exception as encode_error:
                            print(f"Encoding error: {encode_error}")
                            names.append("Encode Error")
                            confidences.append(0.0)
                            person_infos.append({})
                            session_actions.append({'action': 'encode_error'})
                    else:
                        names.append("Align Error")
                        confidences.append(0.0)
                        person_infos.append({})
                        session_actions.append({'action': 'align_error'})

                # Ensure lists have same length as face_boxes
                while len(names) < len(face_boxes):
                    names.append("Error")
                    confidences.append(0.0)
                    person_infos.append({})
                    session_actions.append({'action': 'error'})

            # Draw results with session info
            result_frame = self.draw_results_with_sessions(frame, face_boxes if face_boxes else [],
                                           names, confidences, person_infos, session_actions)

            return result_frame, names, confidences, person_infos

        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, [], [], []

    def _handle_greeting(self, person_name: str, person_info: Dict, session_result: Dict):
        """Handle auto-greeting for a person (session manager đã check cooldown rồi)"""
        try:
            if self.chatbot:
                # Session manager đã quyết định nên chào rồi, chỉ cần tạo greeting text
                greeting = self._generate_greeting_text(person_name, person_info)
                
                print(f"🎯 Auto-greeting {person_name}: {greeting}")
                
                # Speak greeting if TTS is available
                if self.tts_handler and self.tts_handler.available:
                    self.tts_handler.speak_text(greeting)
                
                # Update recognition stats
                if person_name not in self.recognition_stats:
                    self.recognition_stats[person_name] = {
                        'count': 0,
                        'first_seen': time.time(),
                        'last_seen': time.time(),
                        'greetings': 0
                    }
                
                self.recognition_stats[person_name]['greetings'] += 1
                self.recognition_stats[person_name]['last_seen'] = time.time()
            else:
                print(f"⏰ Auto-greeting skipped for {person_name} (chatbot not available)")
                
        except Exception as e:
            print(f"Error handling auto-greeting: {e}")
    
    def _generate_greeting_text(self, person_name: str, person_info: Dict = None) -> str:
        """Generate greeting text without cooldown check (pure text generation)"""
        current_hour = datetime.now().hour
        
        # Chào theo thời gian
        if 5 <= current_hour < 12:
            time_greeting = "Chào buổi sáng"
        elif 12 <= current_hour < 18:
            time_greeting = "Chào buổi chiều"
        else:
            time_greeting = "Chào buổi tối"
        
        # Thêm thông tin cá nhân nếu có
        personal_info = ""
        if person_info:
            if person_info.get('position_applied'):
                personal_info = f" Hôm nay bạn đến phỏng vấn vị trí {person_info['position_applied']} phải không?"
            elif person_info.get('major'):
                personal_info = f" Rất vui được gặp bạn!"
        
        greeting = f"{time_greeting} {person_name}!{personal_info} Tôi là Donkey, trợ lý AI ở đây. Bạn cần hỗ trợ gì không?"
        return greeting
    
    def chat_with_person(self, user_message: str, person_name: str = "") -> str:
        """
        Chat trực tiếp với chatbot (không có cooldown, luôn hoạt động)
        
        Args:
            user_message: Tin nhắn từ người dùng
            person_name: Tên người chat (optional)
            
        Returns:
            Phản hồi từ chatbot
        """
        try:
            if self.chatbot:
                # Add context nếu có person_name
                if person_name and person_name != "Unknown":
                    # Tìm thông tin người từ database
                    person = self.db.get_person(name=person_name)
                    if person:
                        self.chatbot.add_context("Tên người dùng", person_name)
                        self.chatbot.add_context("Thông tin", f"Tuổi: {person.get('age', 'N/A')}, Trường: {person.get('school', 'N/A')}, Chuyên ngành: {person.get('major', 'N/A')}")
                
                # Get response (luôn hoạt động, không có cooldown)
                response = self.chatbot.get_response(user_message, person_name)
                print(f"💬 Chat with {person_name or 'User'}: {user_message[:50]}...")
                print(f"🤖 Response: {response[:100]}...")
                
                return response
            else:
                return "Xin lỗi, hệ thống chatbot hiện không khả dụng."
                
        except Exception as e:
            print(f"Error in chat: {e}")
            return "Xin lỗi, có lỗi xảy ra trong quá trình trò chuyện."
    
    def draw_results_with_sessions(self, frame: np.ndarray, face_boxes: List, 
                                 names: List[str], confidences: List[float], 
                                 person_infos: List[Dict], session_actions: List[Dict]) -> np.ndarray:
        """Draw face recognition results with session information"""
        
        # Start with original draw_results logic
        result_frame = self.draw_results(frame, face_boxes, names, confidences, person_infos)
        
        # Add session information overlay
        try:
            sessions = self.session_manager.get_active_sessions()
            y_offset = 30
            
            # Draw session panel
            cv2.rectangle(result_frame, (10, 10), (400, min(200, 50 + len(sessions) * 20)), (0, 0, 0), -1)
            cv2.rectangle(result_frame, (10, 10), (400, min(200, 50 + len(sessions) * 20)), (255, 255, 255), 2)
            
            cv2.putText(result_frame, f"Active Sessions: {len(sessions)}", 
                       (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw each session
            for person_id, session_info in list(sessions.items())[:5]:  # Show max 5 sessions
                status_color = (0, 255, 0)  # Green for active
                
                if session_info['is_confirming']:
                    status_color = (0, 255, 255)  # Yellow for confirming
                    status_text = f"Confirming ({session_info['confirm_count']}/{self.session_manager.MIN_CONFIRM_FRAMES})"
                elif session_info['has_been_greeted']:
                    if session_info.get('in_cooldown', False):
                        status_color = (0, 165, 255)  # Orange for cooldown
                        status_text = f"Cooldown ({session_info.get('remaining_cooldown_min', 0):.1f}min)"
                    else:
                        status_color = (0, 255, 0)  # Green for active
                        status_text = "Active"
                else:
                    status_text = "Ready"
                
                text = f"{session_info['person_name']}: {status_text}"
                cv2.putText(result_frame, text, (15, 30 + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
                y_offset += 20
            
            # Draw current frame actions
            if session_actions:
                action_y = result_frame.shape[0] - 100
                for i, action in enumerate(session_actions[-3:]):  # Show last 3 actions
                    if 'message' in action:
                        cv2.putText(result_frame, action['message'], 
                                   (10, action_y + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.4, (255, 255, 0), 1)
            
        except Exception as e:
            print(f"Error drawing session info: {e}")
        
        return result_frame

    def run_camera_recognition(self, camera_index: int = 0, save_video: bool = False):
        """
        Run real-time face recognition with camera

        Args:
            camera_index: Camera index (0 for default camera)
            save_video: Whether to save output video
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"Camera {camera_index} opened successfully")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reload from database")
        print("  'c' - Clear all sessions")
        print("  'SPACE' - Chat with Donkey (always available)")
        print("  '+' or '=' - Increase threshold")
        print("  '-' - Decrease threshold")
        print("  's' - Save current frame")
        print("  'i' - Toggle info panel")
        print("  't' - Toggle recognition stats")
        print("  'ESC' - Clear sessions and reset")

        # Video writer setup
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter('face_recognition_db_output.avi', fourcc, 20.0, (640, 480))
            print("Video recording enabled")

        frame_count = 0
        show_info = True
        show_stats = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read frame from camera")
                    break

                frame_count += 1

                # Process frame
                result_frame, names, confidences, person_infos = self.process_frame(frame)

                # Draw info panel if enabled
                if show_info:
                    result_frame = self.draw_info_panel(result_frame)

                # Draw recognition stats if enabled
                if show_stats:
                    result_frame = self.draw_recognition_stats(result_frame)

                # Update FPS
                self.update_fps()

                # Show result
                cv2.imshow('Face Recognition - Database', result_frame)

                # Save video if enabled
                if save_video and video_writer is not None:
                    video_writer.write(result_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("Reloading from database...")
                    try:
                        self.load_known_faces()
                        print("Database reloaded successfully")
                    except Exception as e:
                        print(f"Error reloading database: {e}")
                elif key == ord('+') or key == ord('='):
                    self.similarity_threshold = min(1.0, self.similarity_threshold + 0.02)
                    print(f"Threshold increased to {self.similarity_threshold:.3f}")
                elif key == ord('-'):
                    self.similarity_threshold = max(0.1, self.similarity_threshold - 0.02)
                    print(f"Threshold decreased to {self.similarity_threshold:.3f}")
                elif key == ord('s'):
                    filename = f"recognition_db_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, result_frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('i'):
                    show_info = not show_info
                    print(f"Info panel {'enabled' if show_info else 'disabled'}")
                elif key == ord('t'):
                    show_stats = not show_stats
                    print(f"Recognition stats {'enabled' if show_stats else 'disabled'}")
                elif key == ord('c'):
                    self.session_manager.cleanup_all_sessions()
                    print("All sessions cleared")
                elif key == 27:  # ESC key
                    self.session_manager.cleanup_all_sessions()
                    print("Sessions reset - ready for new recognitions")
                elif key == 32:  # SPACE key for chat
                    print("\n💬 CHAT MODE - Type your message (Enter to send, 'exit' to return):")
                    try:
                        import sys
                        message = input("You: ")
                        if message.strip() and message.lower() != 'exit':
                            # Lấy người cuối cùng được nhận diện (nếu có)
                            last_person = ""
                            if hasattr(self, '_last_recognized_person'):
                                last_person = self._last_recognized_person
                            
                            response = self.chat_with_person(message, last_person)
                            print(f"🤖 Donkey: {response}")
                            
                            # TTS phát phản hồi
                            if self.tts_handler and self.tts_handler.available:
                                self.tts_handler.speak_text(response)
                        else:
                            print("Chat mode exited")
                    except Exception as e:
                        print(f"Chat error: {e}")
                    print("Press any key to continue...")
                    cv2.waitKey(0)

        except KeyboardInterrupt:
            print("\nStopping face recognition...")

        finally:
            # Cleanup
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            print("Camera recognition stopped")

    def recognize_from_image(self, image_path: str) -> Dict:
        """
        Recognize faces from a single image file

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with recognition results
        """
        try:
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                return {'error': f'Cannot read image: {image_path}'}

            # Process frame
            result_frame, names, confidences, person_infos = self.process_frame(frame)

            # Prepare results
            results = {
                'image_path': image_path,
                'faces_detected': len(names),
                'recognitions': []
            }

            for name, confidence, person_info in zip(names, confidences, person_infos):
                recognition = {
                    'name': name,
                    'confidence': confidence,
                    'person_info': person_info
                }
                results['recognitions'].append(recognition)

            # Save result image
            output_path = f"recognition_result_{int(time.time())}.jpg"
            cv2.imwrite(output_path, result_frame)
            results['output_image'] = output_path

            return results

        except Exception as e:
            return {'error': str(e)}


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Real-time Face Recognition with Database')
    parser.add_argument('--database', type=str, default='face_recognition.db',
                        help='Path to SQLite database (default: face_recognition.db)')
    parser.add_argument('--model', type=str, default='../models/arcface.onnx',
                        help='Path to ArcFace ONNX model (default: ../models/arcface.onnx)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Recognition threshold (0.3-0.8, default: 0.6)')
    parser.add_argument('--save-video', action='store_true',
                        help='Save output video')
    parser.add_argument('--image', type=str,
                        help='Recognize faces from image file instead of camera')

    args = parser.parse_args()

    # Validate threshold
    if not 0.1 <= args.threshold <= 1.0:
        print("Warning: Threshold should be between 0.1 and 1.0")
        args.threshold = max(0.1, min(1.0, args.threshold))

    print("=" * 70)
    print("Real-time Face Recognition System (Database)")
    print("=" * 70)
    print(f"Database: {args.database}")
    print(f"Model file: {args.model}")
    print(f"Similarity threshold: {args.threshold}")
    if args.image:
        print(f"Image file: {args.image}")
    else:
        print(f"Camera index: {args.camera}")
        print(f"Save video: {args.save_video}")
    print("=" * 70)

    try:
        # Create face recognition system
        recognition_system = FaceRecognitionSystemDB(
            db_path=args.database,
            model_path=args.model,
            similarity_threshold=args.threshold
        )

        if args.image:
            # Recognize from image
            print(f"Processing image: {args.image}")
            results = recognition_system.recognize_from_image(args.image)

            if 'error' in results:
                print(f"Error: {results['error']}")
            else:
                print(f"\nRecognition Results:")
                print(f"Faces detected: {results['faces_detected']}")
                print(f"Output image saved: {results['output_image']}")

                for i, recognition in enumerate(results['recognitions'], 1):
                    print(f"\nFace {i}:")
                    print(f"  Name: {recognition['name']}")
                    print(f"  Confidence: {recognition['confidence']:.3f}")
                    if recognition['person_info']:
                        for key, value in recognition['person_info'].items():
                            if value and key != 'id':
                                print(f"  {key.title()}: {value}")
        else:
            # Run camera recognition
            recognition_system.run_camera_recognition(
                camera_index=args.camera,
                save_video=args.save_video
            )

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the database file exists")
        print("2. Run encode_faces_db.py first to create face embeddings")
        print("3. Check that the ONNX model file exists at the specified path")
        print("4. Add people to database with add_person.py")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have created the database with face embeddings")
        print("2. Check that your camera is working and not used by other applications")
        print("3. Verify that all model files exist")
        print("4. Install required packages:")
        print("   pip install opencv-python onnxruntime numpy")
        print("5. For GPU support (optional):")
        print("   pip install onnxruntime-gpu")
        print("6. Make sure database contains face embeddings:")
        print("   python add_person.py --stats")
        print("   python encode_faces_db.py")


if __name__ == "__main__":
    main()