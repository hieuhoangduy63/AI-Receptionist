"""
QUẢN LÝ LỊCH HẸN PHỎNG VẤN - Thêm/Sửa/Xóa lịch hẹn và cập nhật từ CSV
"""

import sys
import os
import csv
from database_manager import DatabaseManager
from datetime import datetime, timedelta

class InterviewScheduleManager:
    def __init__(self):
        self.db = DatabaseManager("face_recognition.db")
        
    def get_hr_users(self):
        """Lấy danh sách HR users"""
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, username, full_name FROM hr_users WHERE active = TRUE')
                results = cursor.fetchall()
                return [{'id': row[0], 'username': row[1], 'full_name': row[2]} for row in results]
        except Exception as e:
            print(f"Lỗi lấy danh sách HR: {e}")
            return []

    def add_interview(self):
        """Thêm lịch phỏng vấn mới"""
        print("\n📅 THÊM LỊCH PHỎNG VẤN MỚI")
        print("=" * 50)
        
        # Chọn ứng viên
        print("🔍 Tìm ứng viên:")
        candidates = self.db.list_all_people(active_only=True)
        
        if not candidates:
            print("❌ Không có ứng viên nào trong hệ thống!")
            return False
        
        # Hiển thị danh sách ứng viên
        print(f"\n{'STT':<4} {'ID':<4} {'TÊN':<25} {'VỊ TRÍ ỨNG TUYỂN':<20}")
        print("-" * 60)
        for i, candidate in enumerate(candidates[:10], 1):  # Chỉ hiển thị 10 đầu
            name = candidate['name'][:23] + '..' if len(candidate['name']) > 25 else candidate['name']
            position = candidate.get('position_applied', 'N/A')[:18] + '..' if candidate.get('position_applied') and len(candidate.get('position_applied')) > 20 else candidate.get('position_applied', 'N/A')
            print(f"{i:<4} {candidate['id']:<4} {name:<25} {position:<20}")
        
        if len(candidates) > 10:
            print(f"... và {len(candidates) - 10} ứng viên khác")
        
        # Chọn ứng viên
        print("\nCách chọn ứng viên:")
        print("1. Nhập ID ứng viên")
        print("2. Tìm theo tên")
        
        choice = input("Chọn (1-2): ").strip()
        
        person = None
        if choice == "1":
            try:
                person_id = int(input("Nhập ID ứng viên: "))
                person = self.db.get_person(person_id=person_id)
            except ValueError:
                print("❌ ID phải là số nguyên!")
                return False
        elif choice == "2":
            name = input("Nhập tên ứng viên: ").strip()
            person = self.db.get_person(name=name)
        else:
            print("❌ Lựa chọn không hợp lệ!")
            return False
        
        if not person:
            print("❌ Không tìm thấy ứng viên!")
            return False
        
        print(f"\n✅ Đã chọn ứng viên: {person['name']} (ID: {person['id']})")
        print(f"   💼 Vị trí: {person.get('position_applied', 'N/A')}")
        
        # Chọn HR
        hr_users = self.get_hr_users()
        if not hr_users:
            print("❌ Không có HR user nào trong hệ thống!")
            return False
        
        print(f"\n👤 Chọn HR phụ trách:")
        for i, hr in enumerate(hr_users, 1):
            print(f"{i}. {hr['full_name']} ({hr['username']})")
        
        try:
            hr_choice = int(input("Chọn HR (số thứ tự): ")) - 1
            if hr_choice < 0 or hr_choice >= len(hr_users):
                print("❌ Lựa chọn HR không hợp lệ!")
                return False
            selected_hr = hr_users[hr_choice]
        except ValueError:
            print("❌ Vui lòng nhập số!")
            return False
        
        print(f"✅ Đã chọn HR: {selected_hr['full_name']}")
        
        # Nhập thông tin lịch hẹn
        print(f"\n📋 THÔNG TIN LỊCH HẸN:")
        
        # Ngày giờ phỏng vấn
        print("⏰ Ngày giờ phỏng vấn (định dạng: YYYY-MM-DD HH:MM)")
        print("   Ví dụ: 2025-09-15 09:30")
        interview_date = input("📅 Ngày giờ: ").strip()
        
        # Validate datetime format
        try:
            datetime.strptime(interview_date, '%Y-%m-%d %H:%M')
        except ValueError:
            print("❌ Định dạng ngày giờ không đúng!")
            return False
        
        # Phòng phỏng vấn
        interview_room = input("🏢 Phòng phỏng vấn: ").strip()
        if not interview_room:
            print("❌ Phòng phỏng vấn không được để trống!")
            return False
        
        # Loại phỏng vấn
        print("📋 Loại phỏng vấn:")
        print("1. technical (Kỹ thuật)")
        print("2. hr (Nhân sự)")
        print("3. final (Phỏng vấn cuối)")
        print("4. screening (Sàng lọc)")
        
        interview_types = {
            '1': 'technical',
            '2': 'hr', 
            '3': 'final',
            '4': 'screening'
        }
        
        type_choice = input("Chọn loại phỏng vấn (1-4, mặc định: 1): ").strip() or '1'
        interview_type = interview_types.get(type_choice, 'technical')
        
        # Thời gian (phút)
        duration = input("⏱️ Thời gian phỏng vấn (phút, mặc định: 60): ").strip()
        duration_minutes = int(duration) if duration.isdigit() else 60
        
        # Ghi chú
        notes = input("📝 Ghi chú (tùy chọn): ").strip() or None
        
        try:
            interview_id = self.db.schedule_interview(
                person_id=person['id'],
                hr_user_id=selected_hr['id'],
                interview_date=interview_date,
                interview_room=interview_room,
                interview_type=interview_type,
                duration_minutes=duration_minutes,
                notes=notes
            )
            
            print(f"\n✅ Đã tạo lịch phỏng vấn thành công! (ID: {interview_id})")
            print(f"   👤 Ứng viên: {person['name']}")
            print(f"   👥 HR: {selected_hr['full_name']}")
            print(f"   📅 Thời gian: {interview_date}")
            print(f"   🏢 Phòng: {interview_room}")
            print(f"   📋 Loại: {interview_type}")
            print(f"   ⏱️ Thời gian: {duration_minutes} phút")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Lỗi tạo lịch phỏng vấn: {e}")
            return False

    def list_interviews(self):
        """Hiển thị danh sách lịch phỏng vấn"""
        print("\n📅 DANH SÁCH LỊCH PHỎNG VẤN")
        print("=" * 80)
        
        # Tùy chọn lọc
        print("🔍 Lọc theo:")
        print("1. Tất cả lịch hẹn")
        print("2. Theo ngày (từ-đến)")
        print("3. Theo trạng thái")
        
        choice = input("Chọn (1-3): ").strip()
        
        date_from = date_to = status = None
        
        if choice == "2":
            date_from = input("📅 Từ ngày (YYYY-MM-DD, tùy chọn): ").strip() or None
            date_to = input("📅 Đến ngày (YYYY-MM-DD, tùy chọn): ").strip() or None
        elif choice == "3":
            print("📋 Trạng thái: scheduled, in-progress, completed, cancelled")
            status = input("Nhập trạng thái: ").strip() or None
        
        try:
            interviews = self.db.get_interviews(date_from=date_from, date_to=date_to, status=status)
            
            if not interviews:
                print("❌ Không tìm thấy lịch phỏng vấn nào!")
                return False
            
            print(f"\n📊 Tìm thấy {len(interviews)} lịch phỏng vấn:")
            print(f"\n{'ID':<4} {'ỨNG VIÊN':<20} {'NGÀY GIỜ':<17} {'PHÒNG':<10} {'LOẠI':<12} {'TRẠNG THÁI':<12}")
            print("-" * 85)
            
            for interview in interviews:
                interview_id = interview.get('id', 'N/A')
                name = interview.get('person_name', 'N/A')[:18] + '..' if len(interview.get('person_name', '')) > 20 else interview.get('person_name', 'N/A')
                
                # Format datetime
                interview_date = interview.get('interview_date', 'N/A')
                if interview_date and interview_date != 'N/A':
                    try:
                        dt = datetime.fromisoformat(interview_date.replace('Z', '+00:00'))
                        formatted_date = dt.strftime('%m/%d %H:%M')
                    except:
                        formatted_date = interview_date[:16]
                else:
                    formatted_date = 'N/A'
                
                room = interview.get('interview_room', 'N/A')[:8] + '..' if len(interview.get('interview_room', '')) > 10 else interview.get('interview_room', 'N/A')
                interview_type = interview.get('interview_type', 'N/A')[:10] + '..' if len(interview.get('interview_type', '')) > 12 else interview.get('interview_type', 'N/A')
                status = interview.get('status', 'N/A')[:10] + '..' if len(interview.get('status', '')) > 12 else interview.get('status', 'N/A')
                
                print(f"{interview_id:<4} {name:<20} {formatted_date:<17} {room:<10} {interview_type:<12} {status:<12}")
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi lấy danh sách lịch phỏng vấn: {e}")
            return False

    def update_interview(self):
        """Cập nhật lịch phỏng vấn"""
        print("\n🔄 CẬP NHẬT LỊCH PHỎNG VẤN")
        print("=" * 50)
        
        # Hiển thị danh sách lịch hẹn
        try:
            interviews = self.db.get_interviews()
            
            if not interviews:
                print("❌ Không có lịch phỏng vấn nào!")
                return False
            
            print(f"📋 Danh sách lịch phỏng vấn hiện có:")
            print(f"\n{'ID':<4} {'ỨNG VIÊN':<20} {'NGÀY GIỜ':<17} {'PHÒNG':<10} {'TRẠNG THÁI':<12}")
            print("-" * 70)
            
            for interview in interviews[:10]:  # Hiển thị 10 đầu
                interview_id = interview.get('id', 'N/A')
                name = interview.get('person_name', 'N/A')[:18] + '..' if len(interview.get('person_name', '')) > 20 else interview.get('person_name', 'N/A')
                
                interview_date = interview.get('interview_date', 'N/A')
                if interview_date and interview_date != 'N/A':
                    try:
                        dt = datetime.fromisoformat(interview_date.replace('Z', '+00:00'))
                        formatted_date = dt.strftime('%m/%d %H:%M')
                    except:
                        formatted_date = interview_date[:16]
                else:
                    formatted_date = 'N/A'
                
                room = interview.get('interview_room', 'N/A')[:8] + '..' if len(interview.get('interview_room', '')) > 10 else interview.get('interview_room', 'N/A')
                status = interview.get('status', 'N/A')[:10] + '..' if len(interview.get('status', '')) > 12 else interview.get('status', 'N/A')
                
                print(f"{interview_id:<4} {name:<20} {formatted_date:<17} {room:<10} {status:<12}")
            
            if len(interviews) > 10:
                print(f"... và {len(interviews) - 10} lịch hẹn khác")
            
            # Chọn lịch hẹn cần cập nhật
            try:
                interview_id = int(input("\n📅 Nhập ID lịch phỏng vấn cần cập nhật: "))
                
                # Tìm lịch hẹn
                selected_interview = None
                for interview in interviews:
                    if interview.get('id') == interview_id:
                        selected_interview = interview
                        break
                
                if not selected_interview:
                    print("❌ Không tìm thấy lịch phỏng vấn!")
                    return False
                
                print(f"\n✅ Tìm thấy lịch phỏng vấn:")
                print(f"   👤 Ứng viên: {selected_interview.get('person_name')}")
                print(f"   📅 Thời gian hiện tại: {selected_interview.get('interview_date')}")
                print(f"   🏢 Phòng hiện tại: {selected_interview.get('interview_room')}")
                print(f"   📋 Trạng thái hiện tại: {selected_interview.get('status')}")
                
                # Cập nhật trạng thái
                print(f"\n📋 Trạng thái mới (scheduled, in-progress, completed, cancelled):")
                new_status = input(f"Trạng thái hiện tại [{selected_interview.get('status')}] -> ").strip()
                
                notes = input("📝 Ghi chú mới (tùy chọn): ").strip() or None
                
                if new_status:
                    success = self.db.update_interview_status(interview_id, new_status, notes)
                    if success:
                        print(f"✅ Đã cập nhật trạng thái lịch phỏng vấn thành công!")
                        return True
                    else:
                        print(f"❌ Không thể cập nhật lịch phỏng vấn!")
                        return False
                else:
                    print("⚠️ Không có thông tin nào được cập nhật!")
                    return False
                
            except ValueError:
                print("❌ ID phải là số nguyên!")
                return False
                
        except Exception as e:
            print(f"❌ Lỗi cập nhật lịch phỏng vấn: {e}")
            return False

    def bulk_import_csv(self):
        """Import lịch phỏng vấn từ CSV"""
        print("\n📁 IMPORT LỊCH PHỎNG VẤN TỪ CSV")
        print("=" * 50)
        print("📋 Format file CSV cần có:")
        print("person_name,hr_username,interview_date,interview_room,interview_type,duration_minutes,status,notes")
        print("\n📝 Ví dụ:")
        print("John Doe,admin,2025-09-15 09:00,Room A,technical,60,scheduled,First interview")
        print("Jane Smith,admin,2025-09-15 10:30,Room B,hr,45,scheduled,HR screening")
        
        file_path = input("\n📂 Nhập đường dẫn file CSV: ").strip()
        
        if not os.path.exists(file_path):
            print("❌ File không tồn tại!")
            return False
        
        try:
            successful_imports = 0
            failed_imports = 0
            
            # Lấy danh sách HR users để mapping
            hr_users = self.get_hr_users()
            hr_dict = {hr['username']: hr['id'] for hr in hr_users}
            
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                print(f"\n🔄 Bắt đầu import lịch phỏng vấn...")
                
                for row_num, row in enumerate(reader, start=2):
                    try:
                        person_name = row.get('person_name', '').strip()
                        hr_username = row.get('hr_username', '').strip()
                        interview_date = row.get('interview_date', '').strip()
                        interview_room = row.get('interview_room', '').strip()
                        
                        # Kiểm tra thông tin bắt buộc
                        if not all([person_name, hr_username, interview_date, interview_room]):
                            print(f"❌ Dòng {row_num}: Thiếu thông tin bắt buộc")
                            failed_imports += 1
                            continue
                        
                        # Tìm person_id
                        person = self.db.get_person(name=person_name)
                        if not person:
                            print(f"❌ Dòng {row_num}: Không tìm thấy ứng viên '{person_name}'")
                            failed_imports += 1
                            continue
                        
                        # Tìm hr_user_id
                        hr_user_id = hr_dict.get(hr_username)
                        if not hr_user_id:
                            print(f"❌ Dòng {row_num}: Không tìm thấy HR '{hr_username}'")
                            failed_imports += 1
                            continue
                        
                        # Validate datetime
                        try:
                            datetime.strptime(interview_date, '%Y-%m-%d %H:%M')
                        except ValueError:
                            print(f"❌ Dòng {row_num}: Định dạng ngày giờ không đúng '{interview_date}'")
                            failed_imports += 1
                            continue
                        
                        # Lấy thông tin tùy chọn
                        interview_type = row.get('interview_type', 'technical').strip()
                        duration_str = row.get('duration_minutes', '60').strip()
                        duration_minutes = int(duration_str) if duration_str.isdigit() else 60
                        notes = row.get('notes', '').strip() or None
                        
                        # Tạo lịch phỏng vấn
                        interview_id = self.db.schedule_interview(
                            person_id=person['id'],
                            hr_user_id=hr_user_id,
                            interview_date=interview_date,
                            interview_room=interview_room,
                            interview_type=interview_type,
                            duration_minutes=duration_minutes,
                            notes=notes
                        )
                        
                        print(f"✅ Dòng {row_num}: Tạo lịch cho {person_name} - {interview_date}")
                        successful_imports += 1
                        
                    except Exception as e:
                        print(f"❌ Dòng {row_num}: {e}")
                        failed_imports += 1
            
            print(f"\n📊 KẾT QUẢ IMPORT:")
            print(f"✅ Thành công: {successful_imports}")
            print(f"❌ Thất bại: {failed_imports}")
            
            return successful_imports > 0
            
        except Exception as e:
            print(f"❌ Lỗi đọc file: {e}")
            return False

    def export_to_csv(self):
        """Xuất lịch phỏng vấn ra CSV"""
        print("\n📤 XUẤT LỊCH PHỎNG VẤN RA CSV")
        print("=" * 50)
        
        try:
            interviews = self.db.get_interviews()
            
            if not interviews:
                print("❌ Không có lịch phỏng vấn nào để xuất!")
                return False
            
            filename = f"interviews_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = input(f"📂 Tên file (mặc định: {filename}): ").strip() or filename
            
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                # Header
                writer.writerow(['id', 'person_name', 'hr_name', 'interview_date', 'interview_room', 
                               'interview_type', 'duration_minutes', 'status', 'notes', 'person_email', 'person_phone'])
                
                # Data
                for interview in interviews:
                    writer.writerow([
                        interview.get('id', ''),
                        interview.get('person_name', ''),
                        interview.get('hr_name', ''),
                        interview.get('interview_date', ''),
                        interview.get('interview_room', ''),
                        interview.get('interview_type', ''),
                        interview.get('duration_minutes', ''),
                        interview.get('status', ''),
                        interview.get('notes', ''),
                        interview.get('email', ''),
                        interview.get('phone', '')
                    ])
            
            print(f"✅ Đã xuất {len(interviews)} lịch phỏng vấn ra file: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi xuất file: {e}")
            return False

    def create_sample_csv(self):
        """Tạo file CSV mẫu cho lịch phỏng vấn"""
        print("\n📝 TẠO FILE CSV MẪU CHO LỊCH PHỎNG VẤN")
        print("=" * 50)
        
        filename = input("📂 Tên file mẫu (mặc định: sample_interviews.csv): ").strip() or "sample_interviews.csv"
        
        # Lấy một số ứng viên mẫu
        candidates = self.db.list_all_people(active_only=True)
        
        sample_data = [
            ['person_name', 'hr_username', 'interview_date', 'interview_room', 'interview_type', 'duration_minutes', 'status', 'notes']
        ]
        
        if candidates:
            sample_data.extend([
                [candidates[0]['name'], 'admin', '2025-09-15 09:00', 'Room A', 'technical', '60', 'scheduled', 'Phỏng vấn kỹ thuật đầu tiên'],
                [candidates[1]['name'] if len(candidates) > 1 else 'Jane Smith', 'admin', '2025-09-15 10:30', 'Room B', 'hr', '45', 'scheduled', 'Phỏng vấn HR'],
                [candidates[2]['name'] if len(candidates) > 2 else 'John Doe', 'admin', '2025-09-16 14:00', 'Room C', 'final', '90', 'scheduled', 'Phỏng vấn cuối với ban giám đốc']
            ])
        else:
            sample_data.extend([
                ['Nguyen Van A', 'admin', '2025-09-15 09:00', 'Room A', 'technical', '60', 'scheduled', 'Phỏng vấn kỹ thuật đầu tiên'],
                ['Tran Thi B', 'admin', '2025-09-15 10:30', 'Room B', 'hr', '45', 'scheduled', 'Phỏng vấn HR'],
                ['Le Van C', 'admin', '2025-09-16 14:00', 'Room C', 'final', '90', 'scheduled', 'Phỏng vấn cuối với ban giám đốc']
            ])
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(sample_data)
            
            print(f"✅ Đã tạo file CSV mẫu: {filename}")
            print(f"📋 File chứa {len(sample_data)-1} lịch phỏng vấn mẫu")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi tạo file: {e}")
            return False

def main():
    """Main function"""
    manager = InterviewScheduleManager()
    
    print("📅 QUẢN LÝ LỊCH HẸN PHỎNG VẤN")
    print("=" * 50)
    
    while True:
        print(f"\n🔧 QUẢN LÝ LỊCH HẸN:")
        print("1. 📅 Thêm lịch phỏng vấn mới")
        print("2. 📋 Xem danh sách lịch phỏng vấn")
        print("3. 🔄 Cập nhật trạng thái lịch phỏng vấn")
        print("")
        print("📁 QUẢN LÝ CSV:")
        print("4. 📥 Import từ file CSV")
        print("5. 📤 Export ra file CSV")
        print("6. 📝 Tạo file CSV mẫu")
        print("")
        print("0. 🚪 Thoát")
        
        choice = input("\nNhập lựa chọn (0-6): ").strip()
        
        if choice == "0":
            print("👋 Tạm biệt!")
            break
            
        elif choice == "1":
            manager.add_interview()
            
        elif choice == "2":
            manager.list_interviews()
            
        elif choice == "3":
            manager.update_interview()
            
        elif choice == "4":
            manager.bulk_import_csv()
            
        elif choice == "5":
            manager.export_to_csv()
            
        elif choice == "6":
            manager.create_sample_csv()
            
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()