"""
Enhanced Chatbot with Greeting Cooldown and Fallback Mode
Xử lý chatbot với giới hạn chào lại và chế độ dự phòng
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import time
from typing import Dict, Optional
import tempfile
from gtts import gTTS
import io
import pygame

# Load environment variables
load_dotenv()

class GreetingManager:
    """Quản lý việc chào hỏi với cooldown 5 phút"""
    
    def __init__(self, cooldown_minutes: int = 5):
        self.cooldown_minutes = cooldown_minutes
        self.last_greetings: Dict[str, datetime] = {}  # {person_name: last_greeting_time}
    
    def should_greet(self, person_name: str) -> bool:
        """
        Kiểm tra có nên chào người này không (dựa trên cooldown)
        
        Args:
            person_name: Tên người được nhận diện
            
        Returns:
            True nếu nên chào, False nếu chưa đủ 5 phút
        """
        if not person_name or person_name in ["Unknown", "Error"]:
            return False
            
        now = datetime.now()
        
        # Nếu chưa từng chào person này
        if person_name not in self.last_greetings:
            self.last_greetings[person_name] = now
            return True
        
        # Kiểm tra thời gian từ lần chào cuối
        last_greeting = self.last_greetings[person_name]
        time_diff = now - last_greeting
        
        # Nếu đã quá 5 phút
        if time_diff >= timedelta(minutes=self.cooldown_minutes):
            self.last_greetings[person_name] = now
            return True
        
        return False
    
    def get_time_until_next_greeting(self, person_name: str) -> Optional[int]:
        """
        Lấy số giây còn lại cho đến khi có thể chào lại
        
        Returns:
            Số giây còn lại, hoặc None nếu có thể chào ngay
        """
        if person_name not in self.last_greetings:
            return None
            
        now = datetime.now()
        last_greeting = self.last_greetings[person_name]
        cooldown_end = last_greeting + timedelta(minutes=self.cooldown_minutes)
        
        if now >= cooldown_end:
            return None
        
        return int((cooldown_end - now).total_seconds())


class TTSHandler:
    """Xử lý Text-to-Speech với gTTS đơn giản"""

    def __init__(self):
        try:
            pygame.mixer.init()
            self.available = True
            print("✅ TTS Handler initialized successfully")
        except Exception as e:
            print(f"⚠️ TTS Handler initialization failed: {e}")
            self.available = False

    def speak_text(self, text: str, lang: str = 'vi') -> bool:
        """
        Phát âm văn bản sử dụng gTTS
        
        Args:
            text: Văn bản cần phát âm
            lang: Ngôn ngữ (mặc định 'vi' cho tiếng Việt)
            
        Returns:
            True nếu thành công, False nếu lỗi
        """
        if not self.available or not text.strip():
            return False
            
        try:
            # Tạo file âm thanh tạm thời
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Sử dụng tempfile để tạo file tạm an toàn hơn
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_filename = f"tts_{int(time.time())}.mp3"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            try:
                # Lưu file âm thanh
                tts.save(temp_path)
                
                # Đợi file được tạo hoàn chỉnh
                time.sleep(0.1)
                
                # Kiểm tra file tồn tại
                if not os.path.exists(temp_path):
                    print(f"❌ TTS file not created: {temp_path}")
                    return False
                
                # Phát âm thanh
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Đợi phát xong
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                return True
                
            finally:
                # Cleanup - xóa file tạm
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass  # Ignore cleanup errors
                
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return False


class EnhancedChatBot:
    """Chatbot cải tiến với fallback mode và xử lý lỗi tốt hơn"""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model_name = model
        self.conversation_history = []
        self.context_data = {}
        self.greeting_manager = GreetingManager(cooldown_minutes=5)
        self.fallback_mode = False
        
        # Cấu hình Gemini API
        self._initialize_gemini()
        
        self.enhanced_system_prompt = """
Bạn là một trợ lý AI thông minh, thân thiện và nghiêm túc. Đặc điểm của bạn:

🎯 TÍNH CÁCH:
- Nghiêm túc, cởi mở

💭 CÁCH TRẮC ĐÁP:
- Luôn trả lời bằng tiếng Việt tự nhiên
- Chỉ trả lời ngắn gọn 1-2 câu
- Trả lời ngắn gọn những thông tin cần thiết



🔥 QUAN TRỌNG: Bạn luôn có thể trò chuyện với bất kỳ ai, bất cứ lúc nào! Không có giới hạn thời gian hay cooldown cho việc chat!

Hãy bắt đầu cuộc trò chuyện một cách tự nhiên và thân thiện!
        """

    def _initialize_gemini(self):
        """Khởi tạo Gemini API với xử lý lỗi"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key or api_key == 'your_gemini_api_key_here':
                print("⚠️ GEMINI_API_KEY not found or not set properly. Using fallback mode.")
                self.fallback_mode = True
                return
            
            genai.configure(api_key=api_key)
            
            # Test connection
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
            )
            
            # Test với một câu hỏi đơn giản
            test_response = self.model.generate_content("Hello")
            if test_response and test_response.text:
                print("✅ Gemini API connected successfully")
                self.fallback_mode = False
            else:
                raise Exception("Invalid response from Gemini API")
                
        except Exception as e:
            print(f"⚠️ Gemini API initialization failed: {e}")
            print("🔄 Switching to fallback mode")
            self.fallback_mode = True
            self.model = None

    def add_context(self, key: str, value: str):
        """Thêm thông tin ngữ cảnh"""
        self.context_data[key] = value

    def get_auto_greeting(self, person_name: str, person_info: Dict = None) -> Optional[str]:
        """
        Tạo lời chào tự động khi nhận diện khuôn mặt (với cooldown 5 phút)
        Chỉ dành cho auto-greeting, không ảnh hưởng đến chat thường
        
        Args:
            person_name: Tên người được nhận diện
            person_info: Thông tin chi tiết về người đó
            
        Returns:
            Lời chào nếu nên chào, None nếu không nên chào (chưa đủ 5 phút)
        """
        # Kiểm tra có nên chào không (chỉ áp dụng cho auto-greeting)
        if not self.greeting_manager.should_greet(person_name):
            remaining_seconds = self.greeting_manager.get_time_until_next_greeting(person_name)
            if remaining_seconds:
                remaining_minutes = remaining_seconds // 60
                print(f"⏰ Auto-greeting cooldown for {person_name}. Next greeting in {remaining_minutes}:{remaining_seconds%60:02d}")
            return None
        
        # Tạo lời chào tự động
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
        
        print(f"👋 Auto-greeting generated for {person_name}")
        return greeting

    def get_greeting_message(self, person_name: str, person_info: Dict = None) -> Optional[str]:
        """
        Backward compatibility - chuyển hướng đến get_auto_greeting
        """
        return self.get_auto_greeting(person_name, person_info)

    def get_response(self, user_message: str, person_name: str = "") -> str:
        """
        Lấy phản hồi từ chatbot
        
        Args:
            user_message: Tin nhắn từ người dùng
            person_name: Tên người đang trò chuyện (nếu có)
            
        Returns:
            Phản hồi từ chatbot
        """
        if not user_message.strip():
            return "Bạn có muốn nói gì với tôi không? 😊"
        
        # Nếu trong chế độ fallback
        if self.fallback_mode:
            return self._get_fallback_response(user_message, person_name)
        
        try:
            # Thêm ngữ cảnh nếu có tên người
            context = ""
            if person_name:
                context = f"Tên người đang trò chuyện: {person_name}\n"
                if self.context_data:
                    context += "Thông tin bổ sung:\n"
                    for key, value in self.context_data.items():
                        context += f"- {key}: {value}\n"
            
            # Tạo prompt với ngữ cảnh
            full_prompt = f"{self.enhanced_system_prompt}\n\n{context}\nNgười dùng: {user_message}"
            
            # Gọi Gemini API
            response = self.model.generate_content(full_prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                return self._get_fallback_response(user_message, person_name)
                
        except Exception as e:
            print(f"❌ Gemini API Error: {e}")
            return self._get_fallback_response(user_message, person_name)

    def _get_fallback_response(self, user_message: str, person_name: str = "") -> str:
        """Phản hồi dự phòng khi Gemini API không khả dụng"""
        
        # Phản hồi theo từ khóa
        message_lower = user_message.lower()
        
        # Chào hỏi
        if any(word in message_lower for word in ["xin chào", "chào", "hello", "hi"]):
            if person_name:
                return f"Xin chào {person_name}! Tôi là Donkey, trợ lý AI. Tôi có thể giúp gì cho bạn? 😊"
            return "Xin chào! Tôi là Donkey, trợ lý AI. Rất vui được gặp bạn! 👋"
        
        # Hỏi về tên
        elif any(word in message_lower for word in ["tên", "bạn là ai", "ai"]):
            return "Tôi là Donkey, trợ lý AI thông minh và thân thiện! Được tạo ra để hỗ trợ bạn. 🤖"
        
        # Hỏi về thời gian
        elif any(word in message_lower for word in ["giờ", "thời gian", "bây giờ"]):
            now = datetime.now()
            return f"Bây giờ là {now.strftime('%H:%M')} ngày {now.strftime('%d/%m/%Y')} ⏰"
        
        # Cảm ơn
        elif any(word in message_lower for word in ["cảm ơn", "cám ơn", "thank"]):
            return "Không có gì! Rất vui được giúp đỡ bạn 😊"
        
        # Tạm biệt
        elif any(word in message_lower for word in ["tạm biệt", "bye", "goodbye"]):
            if person_name:
                return f"Tạm biệt {person_name}! Chúc bạn một ngày tốt lành! 👋"
            return "Tạm biệt! Hẹn gặp lại bạn sau! 👋"
        
        # Phỏng vấn
        elif any(word in message_lower for word in ["phỏng vấn", "interview", "lịch hẹn"]):
            return "Tôi có thể giúp bạn tra cứu thông tin lịch phỏng vấn. Bạn cần hỗ trợ gì cụ thể? 📅"
        
        # Mặc định
        else:
            responses = [
                "Tôi hiểu ý bạn rồi! Hiện tại hệ thống AI đang trong chế độ đơn giản. Bạn có thể hỏi tôi về thời gian, lịch phỏng vấn hoặc thông tin cơ bản khác! 😊",
                "Đó là một câu hỏi hay! Tuy nhiên hệ thống AI đang hoạt động ở chế độ cơ bản. Tôi có thể giúp bạn với những thông tin đơn giản! 🤖",
                "Cảm ơn bạn đã chia sẻ! Hiện tại tôi đang hoạt động với những tính năng cơ bản. Bạn cần hỗ trợ gì khác không? ✨"
            ]
            
            # Chọn phản hồi dựa trên thời gian để có tính ngẫu nhiên
            import time
            response_index = int(time.time()) % len(responses)
            return responses[response_index]

    def get_system_status(self) -> Dict:
        """Lấy trạng thái hệ thống chatbot"""
        return {
            "api_available": not self.fallback_mode,
            "model": self.model_name if not self.fallback_mode else "Fallback Mode",
            "greeting_cooldown_minutes": self.greeting_manager.cooldown_minutes,
            "active_conversations": len(self.greeting_manager.last_greetings),
            "last_greetings": {
                name: time.strftime("%H:%M:%S", time.strptime(str(greeting_time), "%Y-%m-%d %H:%M:%S.%f"))
                for name, greeting_time in self.greeting_manager.last_greetings.items()
            } if self.greeting_manager.last_greetings else {}
        }

# Test function
def test_chatbot():
    """Test chatbot functionality"""
    print("🧪 Testing Enhanced Chatbot...")
    
    try:
        chatbot = EnhancedChatBot()
        tts = TTSHandler()
        
        # Test system status
        status = chatbot.get_system_status()
        print(f"📊 System Status: {status}")
        
        # Test greeting with cooldown
        print("\n👋 Testing greeting cooldown...")
        greeting1 = chatbot.get_greeting_message("Test User", {"position_applied": "Developer"})
        print(f"First greeting: {greeting1}")
        
        greeting2 = chatbot.get_greeting_message("Test User")
        print(f"Second greeting (should be None): {greeting2}")
        
        # Test conversation
        print("\n💬 Testing conversation...")
        response = chatbot.get_response("Xin chào!", "Test User")
        print(f"Response: {response}")
        
        # Test TTS if available
        if tts.available:
            print("\n🔊 Testing TTS...")
            tts.speak_text("Xin chào! Tôi là Donkey, trợ lý AI.")
        
        print("\n✅ Chatbot test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chatbot()