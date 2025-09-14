"""
Script để kiểm tra thông tin các ứng viên trong database
"""

import sys
import os
from database_manager import DatabaseManager
from datetime import datetime

def format_date(date_str):
    """Format datetime string for display"""
    if not date_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%d/%m/%Y %H:%M")
    except:
        return date_str

def display_candidate_info(person):
    """Display formatted candidate information"""
    print(f"\n{'='*60}")
    print(f"👤 ID: {person['id']} | Tên: {person['name']}")
    print(f"{'='*60}")
    
    # Thông tin cơ bản
    print(f"📧 Email: {person.get('email', 'N/A')}")
    print(f"📱 Điện thoại: {person.get('phone', 'N/A')}")
    print(f"🎂 Tuổi: {person.get('age', 'N/A')}")
    print(f"🎓 Chuyên ngành: {person.get('major', 'N/A')}")
    print(f"🏫 Trường: {person.get('school', 'N/A')}")
    
    # Thông tin phỏng vấn
    print(f"💼 Vị trí ứng tuyển: {person.get('position_applied', 'N/A')}")
    print(f"📅 Trạng thái phỏng vấn: {person.get('interview_status', 'N/A')}")
    print(f"🏢 Phòng họp: {person.get('meeting_room', 'N/A')}")
    print(f"⏰ Thời gian họp: {person.get('meeting_time', 'N/A')}")
    
    # Ghi chú và thời gian
    if person.get('notes'):
        print(f"📝 Ghi chú: {person['notes']}")
    
    print(f"📅 Tạo lúc: {format_date(person.get('created_at'))}")
    print(f"🔄 Cập nhật lúc: {format_date(person.get('updated_at'))}")
    print(f"✅ Trạng thái: {'Hoạt động' if person.get('active') else 'Không hoạt động'}")

def main():
    """Main function"""
    print("🔍 KIỂM TRA THÔNG TIN ỨNG VIÊN TRONG DATABASE")
    print("=" * 60)
    
    try:
        # Khởi tạo database manager
        db = DatabaseManager("face_recognition.db")
        
        # Lấy danh sách tất cả ứng viên
        candidates = db.list_all_people(active_only=True)
        
        if not candidates:
            print("❌ Không tìm thấy ứng viên nào trong database!")
            return
        
        print(f"📊 Tìm thấy {len(candidates)} ứng viên trong database:")
        
        # Hiển thị danh sách tóm tắt
        print(f"\n{'STT':<4} {'ID':<4} {'TÊN':<20} {'EMAIL':<25} {'VỊ TRÍ':<20}")
        print("-" * 80)
        
        for i, candidate in enumerate(candidates, 1):
            name = candidate['name'][:18] + '..' if len(candidate['name']) > 20 else candidate['name']
            email = candidate.get('email') or 'N/A'
            email = email[:23] + '..' if len(email) > 25 else email
            position = candidate.get('position_applied') or 'N/A'
            position = position[:18] + '..' if len(position) > 20 else position
            
            print(f"{i:<4} {candidate['id']:<4} {name:<20} {email:<25} {position:<20}")
        
        # Tùy chọn xem chi tiết
        print(f"\n{'='*60}")
        print("TÙYCHỌN:")
        print("1. Xem chi tiết tất cả ứng viên")
        print("2. Xem chi tiết ứng viên theo ID")
        print("3. Tìm kiếm ứng viên theo tên")
        print("4. Thống kê database")
        print("0. Thoát")
        
        while True:
            choice = input("\nNhập lựa chọn (0-4): ").strip()
            
            if choice == "0":
                print("👋 Tạm biệt!")
                break
                
            elif choice == "1":
                # Hiển thị chi tiết tất cả
                for candidate in candidates:
                    display_candidate_info(candidate)
                    
            elif choice == "2":
                # Xem chi tiết theo ID
                try:
                    person_id = int(input("Nhập ID ứng viên: "))
                    person = db.get_person(person_id=person_id)
                    if person:
                        display_candidate_info(person)
                    else:
                        print(f"❌ Không tìm thấy ứng viên với ID {person_id}")
                except ValueError:
                    print("❌ ID phải là số nguyên!")
                    
            elif choice == "3":
                # Tìm kiếm theo tên
                name = input("Nhập tên ứng viên: ").strip()
                if name:
                    person = db.get_person(name=name)
                    if person:
                        display_candidate_info(person)
                    else:
                        print(f"❌ Không tìm thấy ứng viên với tên '{name}'")
                        # Tìm kiếm gần đúng
                        similar_candidates = [c for c in candidates if name.lower() in c['name'].lower()]
                        if similar_candidates:
                            print(f"\n🔍 Có thể bạn muốn tìm:")
                            for c in similar_candidates:
                                print(f"  - {c['name']} (ID: {c['id']})")
                                
            elif choice == "4":
                # Thống kê database
                stats = db.get_database_stats()
                print(f"\n📊 THỐNG KÊ DATABASE:")
                print(f"👥 Số ứng viên hoạt động: {stats['active_people']}")
                print(f"🧠 Số face embeddings: {stats['total_embeddings']}")
                print(f"💾 Kích thước database: {stats['database_size_mb']:.2f} MB")
                
                # Thống kê theo trạng thái phỏng vấn
                status_count = {}
                for candidate in candidates:
                    status = candidate.get('interview_status', 'N/A')
                    status_count[status] = status_count.get(status, 0) + 1
                
                print(f"\n📋 Thống kê theo trạng thái phỏng vấn:")
                for status, count in status_count.items():
                    print(f"  - {status}: {count} ứng viên")
                
            else:
                print("❌ Lựa chọn không hợp lệ!")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()