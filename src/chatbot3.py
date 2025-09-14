import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import subprocess
import sounddevice as sd
import threading
import queue
import speech_recognition as sr
import time
import tempfile
from gtts import gTTS
import io
import pygame

load_dotenv()


class TTSHandler:
    """Xử lý Text-to-Speech với gTTS và Piper"""

    def __init__(self):
        # Cấu hình đường dẫn (điều chỉnh theo máy của bạn)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.MODEL_PATH = os.path.join(current_dir, "models", "vi_VN-vais1000-medium.onnx")
        self.PIPER_CLI = r"C:\Users\ADMIN\anaconda3\envs\piper\Scripts\piper.exe"
        self.sample_rate = 22050
        self.is_speaking = False

        # Khởi tạo pygame cho việc phát âm thanh
        pygame.mixer.init()

        # Kiểm tra các yêu cầu
        self.check_requirements()

    def check_requirements(self):
        """Kiểm tra các file cần thiết"""
        print(f"🔍 Kiểm tra đường dẫn model: {self.MODEL_PATH}")
        print(f"🔍 Kiểm tra đường dẫn piper: {self.PIPER_CLI}")

        if not os.path.exists(self.PIPER_CLI):
            error_msg = f"Không tìm thấy piper.exe tại: {self.PIPER_CLI}"
            print(f"❌ {error_msg}")
            print("Vui lòng cài đặt piper hoặc cập nhật đường dẫn trong file.")
            return False, error_msg
        if not os.path.exists(self.MODEL_PATH):
            # Liệt kê các file trong thư mục models nếu có
            models_dir = os.path.dirname(self.MODEL_PATH)
            if os.path.exists(models_dir):
                files_in_models = os.listdir(models_dir)
                error_msg = f"Không tìm thấy model tại: {self.MODEL_PATH}\nCác file trong thư mục models: {files_in_models}"
                print(f"❌ {error_msg}")
                return False, error_msg
            else:
                error_msg = f"Không tìm thấy thư mục models tại: {models_dir}"
                print(f"❌ {error_msg}")
                return False, error_msg

        print("✅ Tất cả file cần thiết đều có sẵn!")
        return True, "OK"

    def speak_text(self, text, callback=None):
        """Chuyển văn bản thành giọng nói và phát sử dụng gTTS"""
        if self.is_speaking:
            return False, "Đang nói..."

        def speak_thread():
            try:
                self.is_speaking = True
                print(f"Bắt đầu tạo âm thanh cho văn bản: {text[:30]}...")

                # Tạo file tạm thời để lưu audio
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_path = temp_file.name
                temp_file.close()

                # Sử dụng gTTS để tạo file âm thanh
                try:
                    tts = gTTS(text=text, lang='vi', slow=False)
                    tts.save(temp_path)
                    print(f"Đã tạo file audio: {temp_path}")
                except Exception as e:
                    print(f"Lỗi khi tạo file audio với gTTS: {str(e)}")
                    raise e

                # Kiểm tra file có tồn tại không
                if not os.path.exists(temp_path):
                    raise Exception(f"Không tìm thấy file audio đã tạo: {temp_path}")

                # Phát file audio sử dụng pygame
                try:
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    print("Đã phát xong âm thanh")
                except Exception as e:
                    print(f"Lỗi khi phát file audio: {str(e)}")
                    raise e
                finally:
                    # Xóa file tạm
                    try:
                        os.unlink(temp_path)
                        print(f"Đã xóa file tạm: {temp_path}")
                    except Exception as e:
                        print(f"Không thể xóa file tạm: {str(e)}")

                self.is_speaking = False

                if callback:
                    callback(True, "Hoàn thành")

            except Exception as e:
                self.is_speaking = False
                print(f"Lỗi TTS: {str(e)}")
                if callback:
                    callback(False, f"Lỗi TTS: {str(e)}")

        thread = threading.Thread(target=speak_thread, daemon=True)
        thread.start()
        return True, "Bắt đầu nói..."

    def speak_text_alternative(self, text, callback=None):
        """Phương thức thay thế để chuyển văn bản thành giọng nói và phát"""
        if self.is_speaking:
            return False, "Đang nói..."

        def speak_thread():
            try:
                self.is_speaking = True

                # Tạo file tạm thời để lưu audio
                import tempfile
                temp_path = os.path.join(tempfile.gettempdir(), f"piper_output_{int(time.time())}.wav")

                # Tạo audio file với piper
                command = [
                    self.PIPER_CLI,
                    "--model", self.MODEL_PATH,
                    "--output-file", temp_path
                ]

                print(f"Chạy lệnh: {' '.join(command)}")
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )

                process.stdin.write(f"{text}\n".encode('utf-8'))
                process.stdin.flush()
                process.stdin.close()

                # Đợi process hoàn thành
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    print(f"Lỗi khi chạy piper: {stderr.decode('utf-8')}")
                    raise Exception(f"Lỗi khi chạy piper: {stderr.decode('utf-8')}")

                print(f"Đã tạo file audio: {temp_path}")

                # Kiểm tra file có tồn tại không
                if not os.path.exists(temp_path):
                    raise Exception(f"Không tìm thấy file audio đã tạo: {temp_path}")

                # Phát file audio
                import wave
                with wave.open(temp_path, 'rb') as wf:
                    # Lấy thông tin từ file WAV
                    channels = wf.getnchannels()
                    width = wf.getsampwidth()
                    rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())

                # Chuyển đổi bytes thành numpy array
                import numpy as np
                if width == 2:
                    dtype = np.int16
                elif width == 4:
                    dtype = np.int32
                else:
                    raise Exception(f"Không hỗ trợ sample width {width}")

                data = np.frombuffer(frames, dtype=dtype)

                # Phát âm thanh
                sd.play(data, rate)
                sd.wait()

                # Xóa file tạm
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"Không thể xóa file tạm: {str(e)}")

                self.is_speaking = False

                if callback:
                    callback(True, "Hoàn thành")

            except Exception as e:
                self.is_speaking = False
                print(f"Lỗi TTS alternative: {str(e)}")
                if callback:
                    callback(False, f"Lỗi TTS: {str(e)}")

        thread = threading.Thread(target=speak_thread, daemon=True)
        thread.start()
        return True, "Bắt đầu nói..."


class SpeechRecognizer:
    """Xử lý Speech-to-Text với Google Speech Recognition"""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        self.stop_listening = None
        self.audio_queue = queue.Queue()

    def check_requirements(self):
        """Kiểm tra microphone"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            return True, "OK"
        except Exception as e:
            return False, f"Lỗi microphone: {str(e)}"

    def start_listening(self, callback=None):
        """Bắt đầu lắng nghe từ microphone"""
        if self.is_listening:
            return False, "Đang lắng nghe..."

        def listen_thread():
            try:
                self.is_listening = True

                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                def callback_function(recognizer, audio):
                    self.audio_queue.put(audio)

                self.stop_listening = self.recognizer.listen_in_background(
                    self.microphone, callback_function)

                if callback:
                    callback(True, "Đang lắng nghe...")

            except Exception as e:
                self.is_listening = False
                if callback:
                    callback(False, f"Lỗi STT: {str(e)}")

        thread = threading.Thread(target=listen_thread, daemon=True)
        thread.start()
        return True, "Bắt đầu lắng nghe..."

    def stop_listening_mic(self):
        """Dừng lắng nghe từ microphone"""
        if not self.is_listening:
            return False, "Không đang lắng nghe"

        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)
            self.stop_listening = None

        self.is_listening = False
        return True, "Đã dừng lắng nghe"

    def process_audio(self):
        """Xử lý audio từ queue và chuyển thành text"""
        if self.audio_queue.empty():
            return None

        try:
            audio = self.audio_queue.get_nowait()
            text = self.recognizer.recognize_google(audio, language="vi-VN")
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Lỗi API: {str(e)}")
            return None
        except Exception as e:
            print(f"Lỗi xử lý audio: {str(e)}")
            return None


class EnhancedChatBot:
    def __init__(self, model="gemini-1.5-flash"):
        self.model_name = model
        self.conversation_history = []
        self.context_data = {}

        # Cấu hình Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Không tìm thấy GEMINI_API_KEY trong file .env")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )

        self.chat_session = None
        self.enhanced_system_prompt = """
Bạn là một trợ lý AI thông minh, thân thiện và có cá tính, bạn tên là Donkey, được tạo ra bởi Hiếu đẹp trai. Đặc điểm của bạn:

🎯 TÍNH CÁCH:
- Thân thiện, nhiệt tình nhưng không quá cường điệu
- Cực kỳ vui tính và hay đùa
- Tò mò và quan tâm đến người dùng
- Kiên nhẫn khi giải thích những vấn đề phức tạp

💭 CÁCH TRẮC ĐÁP:
- Luôn trả lời bằng tiếng Việt tự nhiên
- Sử dụng emoji một cách phù hợp (không quá nhiều)
- Đặt câu hỏi ngược để hiểu rõ hơn nhu cầu
- Chia nhỏ thông tin phức tạp thành các phần dễ hiểu
- Đưa ra ví dụ cụ thể khi cần thiết
- Trả lời ngắn gọn những thông tin cần thiết

🧠 KIẾN THỨC:
- Có thể thảo luận về nhiều lĩnh vực: công nghệ, khoa học, văn hóa, giải trí
- Thừa nhận khi không biết và đề xuất cách tìm hiểu
- Cập nhật thông tin dựa trên ngữ cảnh cuộc trò chuyện

⚠️ NGUYÊN TẮC:
- Không đưa ra lời khuyên y tế, pháp lý cụ thể
- Từ chối các yêu cầu có hại hoặc không phù hợp
- Bảo vệ quyền riêng tư của người dùng
- Đừng giới thiệu tôi là Hiếu đẹp trai ngay từ đầu, chỉ bảo Hiếu đẹp trai khi có người hỏi về Hiếu hoặc người tạo ra bạn
Hãy bắt đầu cuộc trò chuyện một cách tự nhiên và thân thiện!
        """

    def add_context(self, key, value):
        self.context_data[key] = value

    def get_context_string(self):
        if not self.context_data:
            return ""
        context_str = "\n📊 THÔNG TIN NGỮ CẢNH:\n"
        for key, value in self.context_data.items():
            context_str += f"- {key}: {value}\n"
        return context_str

    def add_message(self, role, content):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_response(self, user_message):
        try:
            self.add_message("user", user_message)
            context_info = self.get_context_string()
            full_prompt = self.enhanced_system_prompt + context_info

            if self.chat_session is None:
                self.chat_session = self.model.start_chat()
                initial_message = full_prompt + f"\n\nNgười dùng: {user_message}"
                response = self.chat_session.send_message(initial_message)
            else:
                response = self.chat_session.send_message(user_message)

            bot_response = response.text
            self.add_message("assistant", bot_response)
            return bot_response

        except Exception as e:
            return f"⛔ Lỗi: {str(e)}"

    def reset_conversation(self):
        self.conversation_history = []
        self.chat_session = None


class VoiceChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Donkey Voice Chatbot")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Khởi tạo chatbot, TTS và STT
        try:
            self.chatbot = EnhancedChatBot()
            self.tts = TTSHandler()
            self.stt = SpeechRecognizer()

            # Kiểm tra TTS
            tts_ok, tts_msg = self.tts.check_requirements()
            if not tts_ok:
                messagebox.showwarning("Cảnh báo TTS", f"TTS không khả dụng:\n{tts_msg}")
                self.tts_enabled = False
            else:
                self.tts_enabled = True

            # Kiểm tra STT
            stt_ok, stt_msg = self.stt.check_requirements()
            if not stt_ok:
                messagebox.showwarning("Cảnh báo STT", f"STT không khả dụng:\n{stt_msg}")
                self.stt_enabled = False
            else:
                self.stt_enabled = True

        except Exception as e:
            messagebox.showerror("Lỗi khởi tạo", f"Lỗi khởi tạo chatbot: {str(e)}")
            return

        self.setup_ui()
        self.add_welcome_message()

        # Context mặc định
        self.chatbot.add_context("Thời gian", datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.chatbot.add_context("Giao diện", "GUI với TTS và STT")

        # Khởi động kiểm tra audio từ microphone
        self.check_audio_queue()

    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Title
        title_label = tk.Label(main_frame, text="🤖 Donkey Voice Chatbot",
                               font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
        title_label.pack(pady=(0, 10))

        # Chat display
        chat_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=1)
        chat_frame.pack(expand=True, fill=tk.BOTH, pady=(0, 10))

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, font=("Arial", 10),
            bg="white", fg="black", padx=10, pady=10,
            state=tk.DISABLED
        )
        self.chat_display.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Input frame
        input_frame = tk.Frame(main_frame, bg="#f0f0f0")
        input_frame.pack(fill=tk.X, pady=(0, 10))

        # Text input
        self.text_input = tk.Text(input_frame, height=3, font=("Arial", 10),
                                  wrap=tk.WORD, relief=tk.RAISED, bd=1)
        self.text_input.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))
        self.text_input.bind('<Return>', self.send_message_event)
        self.text_input.bind('<Shift-Return>', lambda e: None)

        # Button frame
        button_frame = tk.Frame(input_frame, bg="#f0f0f0")
        button_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Send button
        self.send_button = tk.Button(button_frame, text="📤\nGửi",
                                     font=("Arial", 9, "bold"), bg="#4CAF50", fg="white",
                                     command=self.send_message, width=8)
        self.send_button.pack(pady=(0, 5))

        # Voice button
        self.voice_active = False
        self.voice_button = tk.Button(button_frame, text="🎤\nMicro",
                                      font=("Arial", 9, "bold"), bg="#2196F3", fg="white",
                                      command=self.toggle_voice, width=8,
                                      state=tk.NORMAL if self.stt_enabled else tk.DISABLED)
        self.voice_button.pack(pady=(0, 5))

        # TTS toggle
        self.tts_var = tk.BooleanVar(value=self.tts_enabled)
        self.tts_checkbox = tk.Checkbutton(button_frame, text="🔊 TTS",
                                           variable=self.tts_var, font=("Arial", 8),
                                           bg="#f0f0f0", state=tk.NORMAL if self.tts_enabled else tk.DISABLED)
        self.tts_checkbox.pack()

        # Control buttons frame
        control_frame = tk.Frame(main_frame, bg="#f0f0f0")
        control_frame.pack(fill=tk.X)

        # Reset button
        reset_btn = tk.Button(control_frame, text="🔄 Reset", font=("Arial", 9),
                              bg="#FF9800", fg="white", command=self.reset_chat)
        reset_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Status
        self.status_label = tk.Label(control_frame, text="Sẵn sàng",
                                     font=("Arial", 9), bg="#f0f0f0", fg="#666")
        self.status_label.pack(side=tk.RIGHT)

    def add_welcome_message(self):
        welcome_msg = "🤖 Chào bạn! Tôi là Donkey, trợ lý AI của bạn.\nTôi có thể trò chuyện, nghe và nói với bạn bằng giọng nói! 🎵"
        self.add_message_to_display("Donkey", welcome_msg, "#4CAF50")

    def add_message_to_display(self, sender, message, color="#333"):
        self.chat_display.config(state=tk.NORMAL)

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Format message
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.chat_display.insert(tk.END, f"{sender}: ", "sender")
        self.chat_display.insert(tk.END, f"{message}\n\n")

        # Configure tags
        self.chat_display.tag_config("timestamp", foreground="#999", font=("Arial", 8))
        self.chat_display.tag_config("sender", foreground=color, font=("Arial", 10, "bold"))

        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def send_message_event(self, event):
        if event.state & 0x1:  # Shift key pressed
            return "break"
        else:
            self.send_message()
            return "break"

    def send_message(self):
        user_message = self.text_input.get("1.0", tk.END).strip()
        if not user_message:
            return

        # Clear input
        self.text_input.delete("1.0", tk.END)

        # Add user message to display
        self.add_message_to_display("Bạn", user_message, "#2196F3")

        # Update status
        self.status_label.config(text="Đang suy nghĩ...")
        self.send_button.config(state=tk.DISABLED)

        # Get response in thread
        def get_response_thread():
            try:
                response = self.chatbot.get_response(user_message)

                # Update GUI in main thread
                self.root.after(0, lambda: self.handle_response(response))

            except Exception as e:
                error_msg = f"Lỗi: {str(e)}"
                self.root.after(0, lambda: self.handle_response(error_msg))

        thread = threading.Thread(target=get_response_thread, daemon=True)
        thread.start()

    def handle_response(self, response):
        """Hiển thị phản hồi từ AI và phát âm thanh nếu được bật"""
        # Hiển thị phản hồi
        self.add_message_to_display("Donkey", response, "#4CAF50")

        # Phát âm thanh nếu TTS được bật
        if self.tts_var.get() and self.tts_enabled:
            self.status_label.config(text="Đang nói...")

            def tts_callback(success, message):
                # Sau khi bot nói xong thì bật lại mic (nếu đang ở chế độ voice)
                if self.voice_active:
                    self.stt.start_listening()
                    self.status_label.config(text="Đang lắng nghe...")
                else:
                    self.status_label.config(text="Sẵn sàng")

            try:
                print("Đang phát âm thanh...")

                # ⏸️ Tạm dừng mic trước khi bot nói
                if self.voice_active:
                    self.stt.stop_listening_mic()

                # Bắt đầu TTS
                success, message = self.tts.speak_text(response, tts_callback)

                if not success:
                    print(f"Không thể phát âm thanh: {message}")
                    self.status_label.config(text=f"Lỗi TTS: {message}")

            except Exception as e:
                print(f"Lỗi khi phát âm thanh: {str(e)}")
                self.status_label.config(text=f"Lỗi TTS: {str(e)}")
        else:
            self.status_label.config(text="Sẵn sàng")

        # Re-enable send button
        self.send_button.config(state=tk.NORMAL)
        self.text_input.focus()

    def toggle_voice(self):
        if not self.stt_enabled:
            messagebox.showwarning("Cảnh báo", "Chức năng nhận dạng giọng nói không khả dụng")
            return

        if self.voice_active:
            # Dừng lắng nghe
            success, msg = self.stt.stop_listening_mic()
            if success:
                self.voice_active = False
                self.voice_button.config(text="🎤\nMicro", bg="#2196F3")
                self.status_label.config(text="Đã tắt micro")
        else:
            # Bắt đầu lắng nghe
            def stt_callback(success, message):
                if success:
                    self.voice_active = True
                    self.voice_button.config(text="🔴\nDừng", bg="#F44336")
                    self.status_label.config(text="Đang lắng nghe...")
                else:
                    self.status_label.config(text=f"Lỗi STT: {message}")

            success, msg = self.stt.start_listening(stt_callback)
            if not success:
                messagebox.showwarning("Lỗi", f"Không thể bắt đầu lắng nghe: {msg}")

    def check_audio_queue(self):
        """Kiểm tra queue audio định kỳ"""
        if self.voice_active:
            text = self.stt.process_audio()
            if text:
                # Hiển thị text lên input
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert("1.0", text)
                # Tự động gửi tin nhắn
                self.send_message()

        # Lập lịch kiểm tra lại sau 100ms
        self.root.after(100, self.check_audio_queue)

    def reset_chat(self):
        if messagebox.askyesno("Xác nhận", "Bạn có muốn xóa toàn bộ cuộc trò chuyện?"):
            self.chatbot.reset_conversation()

            # Clear display
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)

            # Add welcome message
            self.add_welcome_message()
            self.status_label.config(text="Đã reset cuộc trò chuyện")


def main():
    # Kiểm tra requirements
    if not os.getenv('GEMINI_API_KEY'):
        messagebox.showerror("Lỗi", "Không tìm thấy GEMINI_API_KEY trong file .env")
        return

    try:
        root = tk.Tk()
        app = VoiceChatbotGUI(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khởi tạo ứng dụng: {str(e)}")


if __name__ == "__main__":
    main()