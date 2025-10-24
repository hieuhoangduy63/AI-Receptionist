# AI Receptionist System - Hệ thống Lễ tân AI

## Giới thiệu

Hệ thống AI Receptionist là một ứng dụng thông minh tích hợp nhiều công nghệ tiên tiến:

- **Nhận diện khuôn mặt** sử dụng YOLOv8 và ArcFace ONNX model
- **Chatbot AI** với Google Generative AI (Gemini)
- **Text-to-Speech** với gTTS và pygame
- **Phiên làm việc thông minh** với FaceSessionManager
- **Quản lý HR** với cơ sở dữ liệu SQLite
- **Web Interface** với FastAPI và JavaScript

## Cấu trúc dự án

```
TRAE_AI_Receptionist/
├── models/                     # Các model AI
│   ├── arcface.onnx           # Model nhận diện khuôn mặt
│   ├── vi_VN-vais1000-medium.onnx  # Model TTS tiếng Việt
│   └── yolov8x-face-lindevs.pt     # Model phát hiện khuôn mặt
├── src/                       # Mã nguồn chính
│   ├── api.py                 # FastAPI server
│   ├── chatbot3.py            # Chatbot và TTS
│   ├── database_manager.py    # Quản lý cơ sở dữ liệu
│   ├── encode_faces_db.py     # Encode khuôn mặt vào DB
│   ├── encode_faces.py        # Encode khuôn mặt (không DB)
│   ├── face_detect.py         # Phát hiện khuôn mặt
│   ├── face_align.py          # Căn chỉnh khuôn mặt
│   ├── face_recognition_realtime.py  # Nhận diện real-time
│   ├── face_session_manager.py  # Quản lý phiên nhận diện
│   ├── enhanced_chatbot.py    # Chatbot nâng cao với Gemini
│   ├── candidate_manager.py   # Quản lý ứng viên
│   ├── interview_manager.py   # Quản lý lịch phỏng vấn
│   ├── list_data_names.py     # Công cụ liệt kê dữ liệu
│   ├── face_recognition.db    # Database SQLite
│   ├── data/                  # Dữ liệu training khuôn mặt
│   └── static/                # File web tĩnh
├── requirements.txt           # Dependencies
└── README.md                  
```

## Yêu cầu hệ thống

- **Python**: 3.10 (nếu phiên bản cao hơn sẽ không dùng được mediapipe, có thể sử dụng phương pháp khác)
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **GPU**: Không bắt buộc nhưng khuyến nghị có GPU để tăng tốc
- **Webcam**: Để sử dụng tính năng nhận diện real-time
- **Microphone**: Để sử dụng tính năng speech-to-text

## Cài đặt

### Bước 1: Clone dự án

```bash
git clone https://github.com/hieuhoangduy63/AI-Receptionist.git
cd AI-Receptionist
```

### Bước 2: Tạo Virtual Environment

```bash
# Tạo virtual environment (trong project này tôi dùng python 3.10)
python -m venv venv310

# Kích hoạt virtual environment
# Trên Windows:
venv310\Scripts\activate
# Trên Linux/Mac:
source venv310/bin/activate
```

### Bước 3: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình Environment Variables

Tạo file `.env` trong thư mục gốc:

```env
# Google AI API Key (để sử dụng chatbot)
GEMINI_API_KEY=your_google_api_key_here

# Các cấu hình khác (tùy chọn)
DATABASE_PATH=src/face_recognition.db
MODEL_PATH=models/arcface.onnx
```

### Bước 5: Tải Models (nếu chưa có)

Các model cần thiết:

- `arcface.onnx`: Model nhận diện khuôn mặt

- `yolov8x-face-lindevs.pt`: Model phát hiện khuôn mặt

## Cách chạy dự án

### 1. Khởi động API Server

```bash
# Đảm bảo đã kích hoạt virtual environment
venv310\Scripts\activate

# Chạy server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Server sẽ chạy tại: http://localhost:8000

### 2. Truy cập Web Interface

Mở trình duyệt và truy cập:

- **Trang chính**: http://localhost:8000
- **AI Receptionist**: http://localhost:8000/static/ai_receptionist.html
- **API Documentation**: http://localhost:8000/docs

### 3. Sử dụng các tính năng

#### Thêm người vào hệ thống:

1. Truy cập giao diện web
2. Upload ảnh khuôn mặt
3. Nhập thông tin cá nhân
4. Hệ thống sẽ tự động encode và lưu vào database

#### Nhận diện khuôn mặt:

1. Sử dụng webcam
2. Hệ thống sẽ nhận diện và trả về thông tin người

#### Chatbot AI:

1. Nhập câu hỏi vào giao diện
2. Chatbot sẽ trả lời bằng văn bản và giọng nói
3. Hệ thống tự động chào người mới (có cooldown 5 phút)
4. Chat luôn hoạt động kể cả khi không nhận diện được người

## API Endpoints

### Quản lý người

- `POST /people/` - Thêm người mới
- `GET /people/` - Lấy danh sách người
- `GET /people/{person_id}` - Lấy thông tin một người
- `DELETE /people/{person_id}` - Xóa người

### Nhận diện khuôn mặt

- `POST /recognize/` - Nhận diện từ ảnh upload
- `POST /encode/` - Encode khuôn mặt từ ảnh

### Chatbot

- `POST /chat/` - Chat với AI
- `POST /chat/voice/` - Chat bằng giọng nói

### HR Management

- `POST /hr/users/` - Thêm user HR
- `POST /hr/schedule/` - Lên lịch phỏng vấn
- `GET /hr/schedules/` - Xem lịch phỏng vấn

## Troubleshooting

### Lỗi thường gặp:

1. **ModuleNotFoundError**:

   ```bash
   pip install -r requirements.txt
   ```

2. **RuntimeError: Form data requires 'python-multipart'**:

   ```bash
   pip install python-multipart
   ```

3. **Lỗi ONNX Runtime**:

   - Kiểm tra file model có tồn tại
   - Cài đặt lại onnxruntime: `pip install onnxruntime --upgrade`

4. **Lỗi Audio (PyAudio)**:

   - Windows: Tải và cài đặt từ https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
   - Linux: `sudo apt-get install portaudio19-dev`
   - Mac: `brew install portaudio`

5. **Lỗi Database**:
   ```bash
   # Xóa database cũ và tạo mới
   rm src/face_recognition.db
   python src/database_manager.py
   ```

### Kiểm tra hệ thống:

```bash
# Test API
curl http://localhost:8000

# Test database
sqlite3 src/face_recognition.db ".tables"

# Test dependencies
python -c "import cv2, numpy, onnxruntime, google.generativeai; print('All imports successful')"

# Kiểm tra chatbot
python -c "from src.enhanced_chatbot import EnhancedChatBot; cb = EnhancedChatBot(); print(cb.get_response('Hello'))"
```

## Phát triển

### Thêm tính năng mới:

1. Tạo module trong thư mục `src/`
2. Thêm endpoint vào `api.py`
3. Cập nhật database schema nếu cần
4. Test và document

### Cấu trúc Database:

- `people`: Thông tin ứng viên và người dùng hệ thống
- `face_embeddings`: Vector đặc trưng khuôn mặt (128D)
- `hr_users`: Tài khoản HR
- `interview_schedules`: Lịch phỏng vấn

### Công nghệ chính:

- **YOLOv8**: Phát hiện khuôn mặt
- **ArcFace**: Trích xuất đặc trưng khuôn mặt
- **Gemini AI**: Trò chuyện thông minh với ứng viên
- **gTTS**: Text-to-Speech hỗ trợ tiếng Việt
- **FastAPI**: Backend web RESTful API
- **FaceSessionManager**: Hệ thống quản lý phiên nhận diện thông minh
- **SQLite**: Lưu trữ dữ liệu ứng viên và nhận diện (tạm thời cho lượng người dùng nhỏ)
