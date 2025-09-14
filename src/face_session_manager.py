"""
Face Session Manager - Quản lý session nhận diện khuôn mặt thông minh
Tính năng:
1. Xác nhận ổn định: Cần MIN_CONFIRM_FRAMES liên tiếp mới chào
2. Cooldown cá nhân: Mỗi người có cooldown riêng 
3. Reset khi mất dấu: Nếu không thấy trong LOST_RESET_SEC thì reset
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import threading

@dataclass
class PersonSession:
    """Thông tin session của một người"""
    person_id: str
    person_name: str
    first_seen: float  # timestamp
    last_seen: float   # timestamp
    confirm_count: int = 0  # số frame liên tiếp đã thấy
    has_been_greeted: bool = False  # đã chào chưa
    greeting_time: Optional[float] = None  # thời gian chào lần cuối
    is_active: bool = True  # đang active không


class FaceSessionManager:
    """
    Quản lý session nhận diện khuôn mặt với logic ổn định
    """
    
    def __init__(self, 
                 min_confirm_frames: int = 5,      # Cần 5 frame liên tiếp mới chào
                 cooldown_seconds: int = 300,      # 5 phút cooldown
                 lost_reset_seconds: float = 2.0,  # 2 giây không thấy thì reset
                 max_session_minutes: int = 30):   # Session tối đa 30 phút
        
        self.MIN_CONFIRM_FRAMES = min_confirm_frames
        self.COOLDOWN_SEC = cooldown_seconds  
        self.LOST_RESET_SEC = lost_reset_seconds
        self.MAX_SESSION_SEC = max_session_minutes * 60
        
        # Lưu trữ sessions
        self.active_sessions: Dict[str, PersonSession] = {}  # person_id -> session
        self.frame_history: Dict[str, list] = defaultdict(list)  # person_id -> [timestamps]
        
        # Lock để thread-safe
        self._lock = threading.Lock()
        
        print(f"🎯 FaceSessionManager initialized:")
        print(f"   📊 Min confirm frames: {min_confirm_frames}")
        print(f"   ⏰ Cooldown: {cooldown_seconds}s ({cooldown_seconds//60}min)")
        print(f"   🔄 Lost reset: {lost_reset_seconds}s")
        print(f"   ⌛ Max session: {max_session_minutes}min")

    def process_detection(self, person_id: str, person_name: str, confidence: float = 0.0) -> Dict:
        """
        Xử lý một detection mới
        
        Args:
            person_id: ID của người (có thể là số hoặc string)
            person_name: Tên người
            confidence: Độ tin cậy của detection
            
        Returns:
            Dict với thông tin về việc có nên chào không
            {
                'should_greet': bool,
                'session_info': dict,
                'action': str  # 'new_session', 'continue_session', 'greeting_ready', 'cooldown', 'confirmed'
            }
        """
        with self._lock:
            current_time = time.time()
            person_key = str(person_id)  # Đảm bảo là string
            
            # Cleanup old sessions trước
            self._cleanup_old_sessions(current_time)
            
            # Cập nhật frame history
            self.frame_history[person_key].append(current_time)
            
            # Chỉ giữ lại history trong 10 giây gần nhất
            cutoff_time = current_time - 10.0
            self.frame_history[person_key] = [
                t for t in self.frame_history[person_key] if t > cutoff_time
            ]
            
            # Kiểm tra session hiện tại
            if person_key not in self.active_sessions:
                # Tạo session mới
                session = PersonSession(
                    person_id=person_key,
                    person_name=person_name,
                    first_seen=current_time,
                    last_seen=current_time,
                    confirm_count=1
                )
                self.active_sessions[person_key] = session
                
                return {
                    'should_greet': False,
                    'session_info': self._get_session_info(session),
                    'action': 'new_session',
                    'message': f"🆕 New session started for {person_name}"
                }
            
            else:
                # Cập nhật session hiện có
                session = self.active_sessions[person_key]
                session.last_seen = current_time
                
                # Kiểm tra có bị gián đoạn không
                time_gap = current_time - session.last_seen
                if time_gap > self.LOST_RESET_SEC:
                    # Reset session nếu bị gián đoạn quá lâu
                    session.confirm_count = 1
                    session.has_been_greeted = False
                    session.greeting_time = None
                    
                    return {
                        'should_greet': False,
                        'session_info': self._get_session_info(session),
                        'action': 'session_reset',
                        'message': f"🔄 Session reset for {person_name} (was lost for {time_gap:.1f}s)"
                    }
                
                # Tăng confirm count
                session.confirm_count += 1
                
                # Kiểm tra đã đủ confirm frames chưa
                if not session.has_been_greeted and session.confirm_count >= self.MIN_CONFIRM_FRAMES:
                    # Kiểm tra cooldown
                    if session.greeting_time is None or (current_time - session.greeting_time) >= self.COOLDOWN_SEC:
                        # Sẵn sàng chào!
                        session.has_been_greeted = True
                        session.greeting_time = current_time
                        
                        return {
                            'should_greet': True,
                            'session_info': self._get_session_info(session),
                            'action': 'greeting_ready',
                            'message': f"👋 Ready to greet {person_name} (confirmed {session.confirm_count} frames)"
                        }
                    else:
                        # Vẫn trong cooldown
                        remaining_cooldown = self.COOLDOWN_SEC - (current_time - session.greeting_time)
                        return {
                            'should_greet': False,
                            'session_info': self._get_session_info(session),
                            'action': 'cooldown',
                            'message': f"⏳ {person_name} in cooldown ({remaining_cooldown/60:.1f}min left)"
                        }
                
                elif session.has_been_greeted:
                    # Đã chào rồi, chỉ cập nhật session
                    return {
                        'should_greet': False,
                        'session_info': self._get_session_info(session),
                        'action': 'continue_session',
                        'message': f"💬 Continuing session with {person_name}"
                    }
                
                else:
                    # Vẫn đang confirm
                    return {
                        'should_greet': False,
                        'session_info': self._get_session_info(session),
                        'action': 'confirming',
                        'message': f"🔍 Confirming {person_name} ({session.confirm_count}/{self.MIN_CONFIRM_FRAMES})"
                    }

    def mark_person_left(self, person_id: str) -> bool:
        """
        Đánh dấu người đã rời đi
        
        Args:
            person_id: ID của người
            
        Returns:
            True nếu session đã được xóa
        """
        with self._lock:
            person_key = str(person_id)
            if person_key in self.active_sessions:
                session = self.active_sessions[person_key]
                print(f"👋 {session.person_name} has left (session duration: {time.time() - session.first_seen:.1f}s)")
                del self.active_sessions[person_key]
                
                # Xóa frame history
                if person_key in self.frame_history:
                    del self.frame_history[person_key]
                
                return True
            return False

    def get_active_sessions(self) -> Dict[str, Dict]:
        """Lấy danh sách tất cả session đang active"""
        with self._lock:
            current_time = time.time()
            result = {}
            
            for person_id, session in self.active_sessions.items():
                result[person_id] = self._get_session_info(session, current_time)
                
            return result

    def cleanup_all_sessions(self):
        """Xóa tất cả sessions"""
        with self._lock:
            self.active_sessions.clear()
            self.frame_history.clear()
            print("🧹 All sessions cleared")

    def _cleanup_old_sessions(self, current_time: float):
        """Xóa các session quá cũ"""
        to_remove = []
        
        for person_id, session in self.active_sessions.items():
            # Xóa nếu quá lâu không thấy
            if (current_time - session.last_seen) > (self.LOST_RESET_SEC * 5):  # 5x lost reset time
                to_remove.append(person_id)
            # Hoặc nếu session quá lâu (quá MAX_SESSION_SEC)
            elif (current_time - session.first_seen) > self.MAX_SESSION_SEC:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            session = self.active_sessions[person_id]
            print(f"🗑️ Cleaning up old session for {session.person_name}")
            del self.active_sessions[person_id]
            if person_id in self.frame_history:
                del self.frame_history[person_id]

    def _get_session_info(self, session: PersonSession, current_time: Optional[float] = None) -> Dict:
        """Lấy thông tin chi tiết của session"""
        if current_time is None:
            current_time = time.time()
            
        session_duration = current_time - session.first_seen
        time_since_last_seen = current_time - session.last_seen
        
        info = {
            'person_id': session.person_id,
            'person_name': session.person_name,
            'session_duration_sec': round(session_duration, 2),
            'session_duration_min': round(session_duration / 60, 2),
            'confirm_count': session.confirm_count,
            'has_been_greeted': session.has_been_greeted,
            'time_since_last_seen': round(time_since_last_seen, 2),
            'is_confirming': session.confirm_count < self.MIN_CONFIRM_FRAMES,
            'is_ready_to_greet': (session.confirm_count >= self.MIN_CONFIRM_FRAMES and not session.has_been_greeted)
        }
        
        # Thêm thông tin cooldown nếu đã chào
        if session.greeting_time:
            time_since_greeting = current_time - session.greeting_time
            remaining_cooldown = max(0, self.COOLDOWN_SEC - time_since_greeting)
            info.update({
                'time_since_greeting_sec': round(time_since_greeting, 2),
                'time_since_greeting_min': round(time_since_greeting / 60, 2),
                'remaining_cooldown_sec': round(remaining_cooldown, 2),
                'remaining_cooldown_min': round(remaining_cooldown / 60, 2),
                'in_cooldown': remaining_cooldown > 0
            })
        
        return info

    def get_statistics(self) -> Dict:
        """Lấy thống kê về session manager"""
        with self._lock:
            current_time = time.time()
            
            stats = {
                'active_sessions_count': len(self.active_sessions),
                'total_frame_history_count': sum(len(history) for history in self.frame_history.values()),
                'config': {
                    'min_confirm_frames': self.MIN_CONFIRM_FRAMES,
                    'cooldown_seconds': self.COOLDOWN_SEC,
                    'lost_reset_seconds': self.LOST_RESET_SEC,
                    'max_session_seconds': self.MAX_SESSION_SEC
                },
                'sessions': {}
            }
            
            for person_id, session in self.active_sessions.items():
                stats['sessions'][person_id] = self._get_session_info(session, current_time)
            
            return stats


# Test function
def test_session_manager():
    """Test face session manager"""
    print("🧪 Testing Face Session Manager...")
    
    manager = FaceSessionManager(
        min_confirm_frames=3,  # Để test nhanh
        cooldown_seconds=10,   # 10 giây để test
        lost_reset_seconds=1.0 # 1 giây để test
    )
    
    # Test case 1: Người mới
    print("\n📝 Test 1: New person detection")
    for i in range(5):
        result = manager.process_detection("person_1", "John Doe", 0.95)
        print(f"Frame {i+1}: {result['action']} - {result['message']}")
        time.sleep(0.1)
    
    # Test case 2: Cooldown
    print("\n📝 Test 2: Cooldown test")
    time.sleep(1)
    result = manager.process_detection("person_1", "John Doe", 0.95)
    print(f"After cooldown: {result['action']} - {result['message']}")
    
    # Test case 3: Người mới khác
    print("\n📝 Test 3: Different person")
    for i in range(4):
        result = manager.process_detection("person_2", "Jane Smith", 0.90)
        print(f"Frame {i+1}: {result['action']} - {result['message']}")
        time.sleep(0.1)
    
    # Stats
    print("\n📊 Final Statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        if key != 'sessions':
            print(f"  {key}: {value}")
    
    print("\n✅ Session Manager test completed!")


if __name__ == "__main__":
    test_session_manager()