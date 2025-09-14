import os
import numpy as np
import cv2
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .database_manager import DatabaseManager
from .encode_faces_db import FaceEncoderDB
from .face_recognition_realtime import FaceRecognitionSystemDB
from .enhanced_chatbot import EnhancedChatBot, TTSHandler

# ================== CONFIG ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "face_recognition.db")
MODEL_PATH = os.path.join(BASE_DIR, "../models/arcface.onnx")

# ================== INIT APP ==================
app = FastAPI(title="AI Receptionist API with HR Management")

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Initialize components (lazy loading)
db = None
encoder = None
recognizer = None
chatbot = None
tts_handler = None

def get_db():
    global db
    if db is None:
        db = DatabaseManager(DB_PATH)
    return db

def get_encoder():
    global encoder
    if encoder is None:
        encoder = FaceEncoderDB(MODEL_PATH, DB_PATH)
    return encoder

def get_recognizer():
    global recognizer
    if recognizer is None:
        recognizer = FaceRecognitionSystemDB(DB_PATH, MODEL_PATH)
    return recognizer

def get_chatbot():
    global chatbot
    if chatbot is None:
        try:
            chatbot = EnhancedChatBot()
        except Exception as e:
            print(f"Warning: Could not initialize chatbot: {e}")
            chatbot = None
    return chatbot

def get_tts_handler():
    global tts_handler
    if tts_handler is None:
        try:
            tts_handler = TTSHandler()
        except Exception as e:
            print(f"Warning: Could not initialize TTS: {e}")
            tts_handler = None
    return tts_handler


# ================== PYDANTIC MODELS ==================
class LoginRequest(BaseModel):
    username: str
    password: str


class CandidateCreate(BaseModel):
    name: str
    age: Optional[int] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    school: Optional[str] = None
    major: Optional[str] = None
    position_applied: Optional[str] = None
    notes: Optional[str] = None


class CandidateUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    school: Optional[str] = None
    major: Optional[str] = None
    position_applied: Optional[str] = None
    interview_status: Optional[str] = None
    notes: Optional[str] = None


class InterviewSchedule(BaseModel):
    person_id: int
    interview_date: str  # ISO format datetime
    interview_room: str
    interview_type: Optional[str] = "technical"
    duration_minutes: Optional[int] = 60
    notes: Optional[str] = None


class InterviewStatusUpdate(BaseModel):
    status: str
    notes: Optional[str] = None


# ================== MAIN ROUTES ==================
@app.get("/")
def root():
    """Serve main interface"""
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.get("/ai_receptionist")
def receptionist():
    """Serve Receptionist interface"""
    return FileResponse(os.path.join(BASE_DIR, "static", "ai_receptionist.html"))

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """Face recognition endpoint"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    frame, names, confidences, infos = get_recognizer().process_frame(image)

    recognitions = []
    for name, conf, info in zip(names, confidences, infos):
        person_dict = {
            "name": name,
            "confidence": round(conf, 3),
            "info": info
        }

        # ✅ Gọi update_checkin khi nhận diện đúng ứng viên
        if name not in ["Unknown", "Error"] and info and "id" in info:
            get_db().update_checkin(info["id"])
            person_dict["checkin_status"] = "checked_in"
            person_dict["checkin_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            person_dict["checkin_status"] = "not_checked_in"
            person_dict["checkin_time"] = None

        recognitions.append(person_dict)

    return {"faces_detected": len(recognitions), "recognitions": recognitions}



# ================== HR AUTHENTICATION ==================
@app.post("/login")
async def login(request: LoginRequest):
    """HR user login"""
    try:
        user = get_db().verify_hr_user(request.username, request.password)

        if user:
            return {
                "success": True,
                "message": "Đăng nhập thành công",
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "full_name": user["full_name"],
                    "role": user["role"]
                }
            }
        else:
            return {
                "success": False,
                "message": "Tên đăng nhập hoặc mật khẩu không đúng"
            }

    except Exception as e:
        return {
            "success": False,
            "message": f"Lỗi đăng nhập: {str(e)}"
        }


# ================== HR MANAGEMENT ROUTES ==================
@app.get("/hr/stats")
async def get_hr_stats():
    """Get dashboard statistics"""
    try:
        stats = get_db().get_dashboard_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hr/candidates")
async def get_candidates():
    """Get all candidates"""
    try:
        candidates = get_db().list_all_people()
        return candidates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hr/candidates")
async def create_candidate(candidate: CandidateCreate):
    """Create new candidate"""
    try:
        person_id = get_db().add_person(
            name=candidate.name,
            age=candidate.age,
            email=candidate.email,
            phone=candidate.phone,
            school=candidate.school,
            major=candidate.major,
            position_applied=candidate.position_applied,
            notes=candidate.notes
        )

        return {
            "success": True,
            "message": "Ứng viên đã được thêm thành công",
            "person_id": person_id
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/hr/candidates/{person_id}")
async def update_candidate(person_id: int, candidate: CandidateUpdate):
    """Update candidate information"""
    try:
        # Convert to dict and remove None values
        update_data = {k: v for k, v in candidate.dict().items() if v is not None}

        success = get_db().update_person(person_id=person_id, **update_data)

        if success:
            return {
                "success": True,
                "message": "Thông tin ứng viên đã được cập nhật"
            }
        else:
            raise HTTPException(status_code=404, detail="Không tìm thấy ứng viên")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hr/candidates/{person_id}")
async def get_candidate(person_id: int):
    """Get specific candidate"""
    try:
        candidate = get_db().get_person(person_id=person_id)
        if candidate:
            return candidate
        else:
            raise HTTPException(status_code=404, detail="Không tìm thấy ứng viên")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/hr/candidates/{person_id}")
async def delete_candidate(person_id: int, soft_delete: bool = True):
    """Delete candidate (soft delete by default)"""
    try:
        success = get_db().delete_person(person_id=person_id, soft_delete=soft_delete)

        if success:
            return {
                "success": True,
                "message": "Ứng viên đã được xóa" if not soft_delete else "Ứng viên đã được vô hiệu hóa"
            }
        else:
            raise HTTPException(status_code=404, detail="Không tìm thấy ứng viên")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================== INTERVIEW MANAGEMENT ==================
@app.get("/hr/interviews")
async def get_interviews(
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        status: Optional[str] = None
):
    """Get interview schedules with optional filtering"""
    try:
        interviews = get_db().get_interviews(date_from, date_to, status)
        return interviews
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hr/interviews")
async def schedule_interview(schedule: InterviewSchedule):
    """Schedule a new interview"""
    try:
        # Get current HR user ID (in real app, this would come from authentication)
        hr_user_id = 1  # Default admin user

        interview_id = get_db().schedule_interview(
            person_id=schedule.person_id,
            hr_user_id=hr_user_id,
            interview_date=schedule.interview_date,
            interview_room=schedule.interview_room,
            interview_type=schedule.interview_type,
            duration_minutes=schedule.duration_minutes,
            notes=schedule.notes
        )

        return {
            "success": True,
            "message": "Lịch phỏng vấn đã được tạo thành công",
            "interview_id": interview_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/hr/interviews/{interview_id}/status")
async def update_interview_status(interview_id: int, status_update: InterviewStatusUpdate):
    """Update interview status"""
    try:
        success = get_db().update_interview_status(
            interview_id=interview_id,
            status=status_update.status,
            notes=status_update.notes
        )

        if success:
            return {
                "success": True,
                "message": "Trạng thái phỏng vấn đã được cập nhật"
            }
        else:
            raise HTTPException(status_code=404, detail="Không tìm thấy lịch phỏng vấn")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hr/interviews/today")
async def get_today_interviews():
    """Get today's interviews"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        interviews = get_db().get_interviews(
            date_from=today + " 00:00:00",
            date_to=today + " 23:59:59"
        )
        return {"success": True, "interviews": interviews}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/chatbot/speak")
async def chatbot_speak(request: dict):
    """Make chatbot speak a message"""
    try:
        message = request.get("message", "")
        if not message:
            return {"success": False, "message": "Không có tin nhắn để nói"}
        
        tts_handler = get_tts_handler()
        if tts_handler:
            # Use TTS to speak the message
            tts_handler.speak_text(message)
            return {"success": True, "message": "Đã phát âm thanh thành công"}
        else:
            return {"success": False, "message": "TTS không khả dụng"}
            
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/chatbot/chat")
async def chatbot_chat(request: dict):
    """Chat with the AI assistant (always available, no restrictions)"""
    try:
        user_message = request.get("message", "")
        person_name = request.get("person_name", "")  # Optional, chỉ để context
        
        if not user_message:
            return {"success": False, "message": "Không có tin nhắn"}

        chatbot = get_chatbot()
        if not chatbot:
            return {"success": False, "message": "Hệ thống chatbot hiện không khả dụng"}
        
        # Optional: Add context if person_name provided (không bắt buộc)
        if person_name and person_name.strip() and person_name != "Unknown":
            try:
                person = get_db().get_person(name=person_name)
                if person:
                    chatbot.add_context("Tên người dùng", person_name)
                    chatbot.add_context("Thông tin", f"Tuổi: {person.get('age', 'N/A')}, Trường: {person.get('school', 'N/A')}, Chuyên ngành: {person.get('major', 'N/A')}")
            except:
                pass  # Ignore errors, chat should work regardless
        
        # Chat luôn hoạt động, không có bất kỳ giới hạn nào
        response = chatbot.get_response(user_message, person_name)
        
        return {
            "success": True,
            "response": response,
            "person_name": person_name,
            "chat_mode": "unrestricted",
            "note": "Chat always available - no cooldown, no recognition required"
        }
            
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/chatbot/greeting")
async def get_greeting(request: dict):
    """Get personalized greeting with cooldown (5 minutes)"""
    try:
        person_name = request.get("person_name", "")
        
        if not person_name:
            return {"success": False, "message": "Cần tên người dùng"}
        
        chatbot = get_chatbot()
        if not chatbot:
            return {"success": False, "message": "Hệ thống chatbot không khả dụng"}
        
        # Get person info for personalized greeting
        person_info = None
        person = get_db().get_person(name=person_name)
        if person:
            person_info = {
                "position_applied": person.get("position_applied"),
                "major": person.get("major"),
                "school": person.get("school")
            }
            
            # Add context to chatbot
            chatbot.add_context("Tên người dùng", person_name)
            chatbot.add_context("Thông tin", f"Tuổi: {person.get('age', 'N/A')}, Trường: {person.get('school', 'N/A')}, Chuyên ngành: {person.get('major', 'N/A')}")
        
        # Get greeting with cooldown check (chỉ cho auto-greeting)
        greeting = chatbot.get_auto_greeting(person_name, person_info)
        
        if greeting:
            return {
                "success": True,
                "greeting": greeting,
                "person_name": person_name,
                "should_greet": True
            }
        else:
            # Check remaining time
            remaining_seconds = chatbot.greeting_manager.get_time_until_next_greeting(person_name)
            return {
                "success": True,
                "greeting": None,
                "person_name": person_name,
                "should_greet": False,
                "remaining_seconds": remaining_seconds,
                "message": f"Đã chào {person_name} gần đây. Sẽ chào lại sau {remaining_seconds//60}:{remaining_seconds%60:02d}"
            }
            
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/chatbot/status")
async def get_chatbot_status():
    """Get chatbot system status"""
    try:
        chatbot = get_chatbot()
        if chatbot:
            status = chatbot.get_system_status()
            return {"success": True, "status": status}
        else:
            return {
                "success": False, 
                "status": {
                    "api_available": False,
                    "model": "Not Available",
                    "greeting_cooldown_minutes": 5,
                    "active_conversations": 0,
                    "last_greetings": {}
                }
            }
            
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/get_interview_info/{person_name}")
async def get_interview_info(person_name: str):
    """Get interview information for a specific person"""
    try:
        # Get person info
        person = get_db().get_person(name=person_name)
        if not person:
            return {"success": False, "message": "Không tìm thấy thông tin ứng viên"}
        
        # Get today's interview for this person
        today = datetime.now().strftime('%Y-%m-%d')
        interviews = get_db().get_interviews(
            date_from=today + " 00:00:00",
            date_to=today + " 23:59:59"
        )
        
        person_interview = None
        for interview in interviews:
            if interview['person_name'] == person_name:
                person_interview = interview
                break
        
        if person_interview:
            # Format interview time
            interview_datetime = datetime.fromisoformat(person_interview['interview_date'])
            interview_time = interview_datetime.strftime('%H:%M')
            
            return {
                "success": True,
                "person": person,
                "interview": {
                    "time": interview_time,
                    "room": person_interview['interview_room'],
                    "type": person_interview['interview_type'],
                    "duration": person_interview['duration_minutes']
                },
                "greeting_message": f"Xin chào {person_name}, hôm nay bạn có lịch phỏng vấn lúc {interview_time} tại {person_interview['interview_room']}"
            }
        else:
            return {
                "success": True,
                "person": person,
                "interview": None,
                "greeting_message": f"Xin chào {person_name}, chào mừng bạn đến với văn phòng!"
            }
            
    except Exception as e:
        return {"success": False, "message": str(e)}


# ================== FACE ENROLLMENT ROUTES ==================
@app.post("/enroll-upload")
async def enroll_upload(
        name: str,
        files: List[UploadFile] = File(...),
        age: int = None,
        major: str = None,
        school: str = None,
        phone: str = None,
        email: str = None,
        position_applied: str = None,
        meeting_room: str = None,
        meeting_time: str = None
):
    """Enroll new candidate with face images (enhanced from original)"""
    all_embeddings = []
    total_faces = 0

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            continue

        # Detect and align faces
        recognizer = get_recognizer()
        face_boxes, _, _ = recognizer.detector.detect_faces(image)
        aligned_faces = recognizer.aligner.align_faces_from_image(image, face_boxes)

        # Encode faces
        for face in aligned_faces:
            emb = recognizer.encoder.encode_face(face)
            if emb is not None:
                all_embeddings.append(emb)
                total_faces += 1

    if not all_embeddings:
        return JSONResponse(
            content={"status": "failed", "reason": "No valid faces found"},
            status_code=400
        )

    try:
        # Calculate average embedding
        avg_emb = np.mean(all_embeddings, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)

        # Add person to database with extended information
        person_id = get_db().add_person(
            name=name,
            age=age,
            major=major,
            school=school,
            phone=phone,
            email=email,
            position_applied=position_applied,
            meeting_room=meeting_room,
            meeting_time=meeting_time
        )

        # Save face embedding
        get_db().save_face_embedding(person_id, avg_emb, faces_processed=total_faces)

        return {
            "status": "success",
            "person_id": person_id,
            "faces": total_faces,
            "message": f"Đã đăng ký thành công {name} với {total_faces} khuôn mặt"
        }

    except ValueError as e:
        return JSONResponse(
            content={"status": "failed", "reason": str(e)},
            status_code=400
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "failed", "reason": f"Database error: {str(e)}"},
            status_code=500
        )


# ================== UTILITY ROUTES ==================
@app.get("/people")
def list_people():
    """List all people in database (unchanged from original)"""
    return get_db().list_all_people()


@app.get("/stats")
def stats():
    """Get database statistics (unchanged from original)"""
    return get_db().get_database_stats()


@app.post("/hr/reload-faces")
async def reload_face_database():
    """Reload face embeddings from database"""
    try:
        get_recognizer().load_known_faces()
        return {
            "success": True,
            "message": "Cơ sở dữ liệu khuôn mặt đã được tải lại"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================== ERROR HANDLERS ==================
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Không tìm thấy tài nguyên yêu cầu"}
    )


@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Lỗi máy chủ nội bộ"}
    )


# ================== MAIN ==================
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🤖 AI Receptionist System with HR Management")
    print("=" * 60)
    print(f"📁 Database: {DB_PATH}")
    print(f"🧠 Model: {MODEL_PATH}")
    print("🌐 Server sẽ chạy tại:")
    print("   👉 http://localhost:8000 (Click để mở)")
    print("   👉 http://localhost:8000/ai_receptionist (AI Receptionist)")
    print("=" * 60)
    print("🔑 Default HR Login:")
    print("   Username: admin")
    print("   Password: admin123")
    print("=" * 60)

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)