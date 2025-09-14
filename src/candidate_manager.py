"""
QUẢN LÝ ỨNG VIÊN - Thêm/Sửa/Xóa ứng viên và cập nhật từ CSV
"""

import sys
import os
import csv
from database_manager import DatabaseManager
from datetime import datetime

class CandidateManager:
    def __init__(self):
        self.db = DatabaseManager("face_recognition.db")
    
    def add_candidate(self):
        """Thêm ứng viên mới"""
        print("\n🆕 THÊM ỨNG VIÊN MỚI")
        print("=" * 50)
        
        # Nhập thông tin cơ bản
        name = input("👤 Tên ứng viên (bắt buộc): ").strip()
        if not name:
            print("❌ Tên không được để trống!")
            return False
        
        print("\n📋 THÔNG TIN CÁ NHÂN:")
        age = input("🎂 Tuổi (tùy chọn): ").strip()
        age = int(age) if age.isdigit() else None
        
        email = input("📧 Email (tùy chọn): ").strip() or None
        phone = input("📱 Số điện thoại (tùy chọn): ").strip() or None
        
        print("\n🎓 THÔNG TIN HỌC TẬP:")
        major = input("📚 Chuyên ngành (tùy chọn): ").strip() or None
        school = input("🏫 Trường (tùy chọn): ").strip() or None
        
        print("\n💼 THÔNG TIN CÔNG VIỆC:")
        position_applied = input("🎯 Vị trí ứng tuyển (tùy chọn): ").strip() or None
        
        print("\n📅 THÔNG TIN PHỎNG VẤN:")
        print("Trạng thái phỏng vấn có thể: scheduled, in-progress, completed, hired, rejected")
        interview_status = input("📋 Trạng thái phỏng vấn (mặc định: scheduled): ").strip() or 'scheduled'
        
        print("\n📝 THÔNG TIN KHÁC:")
        notes = input("📋 Ghi chú (tùy chọn): ").strip() or None
        
        try:
            person_id = self.db.add_person(
                name=name,
                age=age,
                email=email,
                phone=phone,
                major=major,
                school=school,
                position_applied=position_applied,
                interview_status=interview_status,
                notes=notes
            )
            
            print(f"\n✅ Đã thêm ứng viên '{name}' thành công! (ID: {person_id})")
            return True
            
        except ValueError as e:
            print(f"\n❌ Lỗi: {e}")
            return False
        except Exception as e:
            print(f"\n❌ Lỗi không xác định: {e}")
            return False

    def update_candidate(self):
        """Cập nhật thông tin ứng viên"""
        print("\n🔄 CẬP NHẬT THÔNG TIN ỨNG VIÊN")
        print("=" * 50)
        
        # Chọn cách tìm ứng viên
        print("Tìm ứng viên theo:")
        print("1. ID")
        print("2. Tên")
        
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
        
        print(f"\n✅ Tìm thấy ứng viên: {person['name']} (ID: {person['id']})")
        print("Thông tin hiện tại:")
        print(f"  📧 Email: {person.get('email') or 'N/A'}")
        print(f"  📱 Điện thoại: {person.get('phone') or 'N/A'}")
        print(f"  🎂 Tuổi: {person.get('age') or 'N/A'}")
        print(f"  💼 Vị trí ứng tuyển: {person.get('position_applied') or 'N/A'}")
        print(f"  📋 Trạng thái: {person.get('interview_status') or 'N/A'}")
        
        print(f"\n📝 Nhập thông tin mới (để trống = giữ nguyên):")
        
        # Thu thập thông tin cập nhật
        updates = {}
        
        email = input(f"📧 Email hiện tại: [{person.get('email') or 'N/A'}] -> ").strip()
        if email:
            updates['email'] = email
        
        phone = input(f"📱 Điện thoại hiện tại: [{person.get('phone') or 'N/A'}] -> ").strip()
        if phone:
            updates['phone'] = phone
        
        age = input(f"🎂 Tuổi hiện tại: [{person.get('age') or 'N/A'}] -> ").strip()
        if age.isdigit():
            updates['age'] = int(age)
        
        major = input(f"📚 Chuyên ngành hiện tại: [{person.get('major') or 'N/A'}] -> ").strip()
        if major:
            updates['major'] = major
        
        school = input(f"🏫 Trường hiện tại: [{person.get('school') or 'N/A'}] -> ").strip()
        if school:
            updates['school'] = school
        
        position = input(f"💼 Vị trí ứng tuyển hiện tại: [{person.get('position_applied') or 'N/A'}] -> ").strip()
        if position:
            updates['position_applied'] = position
        
        print("📋 Trạng thái phỏng vấn: scheduled, in-progress, completed, hired, rejected")
        status = input(f"📋 Trạng thái hiện tại: [{person.get('interview_status') or 'scheduled'}] -> ").strip()
        if status:
            updates['interview_status'] = status
        
        notes = input(f"📝 Ghi chú hiện tại: [{person.get('notes') or 'N/A'}] -> ").strip()
        if notes:
            updates['notes'] = notes
        
        if not updates:
            print("⚠️ Không có thông tin nào được cập nhật!")
            return False
        
        try:
            success = self.db.update_person(person_id=person['id'], **updates)
            if success:
                print(f"\n✅ Đã cập nhật thông tin cho '{person['name']}' thành công!")
                
                # Hiển thị thông tin đã cập nhật
                print(f"\n📋 Thông tin sau khi cập nhật:")
                for field, value in updates.items():
                    field_name = {
                        'email': '📧 Email',
                        'phone': '📱 Điện thoại', 
                        'age': '🎂 Tuổi',
                        'major': '📚 Chuyên ngành',
                        'school': '🏫 Trường',
                        'position_applied': '💼 Vị trí ứng tuyển',
                        'interview_status': '📋 Trạng thái',
                        'notes': '📝 Ghi chú'
                    }.get(field, field)
                    print(f"  {field_name}: {value}")
                
                return True
            else:
                print(f"\n❌ Không thể cập nhật thông tin cho '{person['name']}'!")
                return False
                
        except Exception as e:
            print(f"\n❌ Lỗi khi cập nhật: {e}")
            return False

    def delete_candidate(self):
        """Xóa ứng viên (soft delete)"""
        print("\n🗑️ XÓA ỨNG VIÊN")
        print("=" * 50)
        
        # Chọn cách tìm ứng viên
        print("Tìm ứng viên cần xóa theo:")
        print("1. ID")
        print("2. Tên")
        
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
        
        print(f"\n⚠️ Bạn có chắc muốn xóa ứng viên: {person['name']} (ID: {person['id']})?")
        print(f"   📧 Email: {person.get('email') or 'N/A'}")
        print(f"   💼 Vị trí: {person.get('position_applied') or 'N/A'}")
        
        confirm = input("\nNhập 'XAC NHAN' để xóa (hoặc Enter để hủy): ").strip()
        
        if confirm.upper() != 'XAC NHAN':
            print("❌ Đã hủy thao tác xóa!")
            return False
        
        try:
            # Soft delete using dedicated method
            success = self.db.delete_person(person_id=person['id'], soft_delete=True)
            if success:
                print(f"✅ Đã xóa ứng viên '{person['name']}' thành công!")
                return True
            else:
                print(f"❌ Không thể xóa ứng viên '{person['name']}'!")
                return False
                
        except Exception as e:
            print(f"❌ Lỗi khi xóa: {e}")
            return False

    def bulk_import_csv(self):
        """Cập nhật hàng loạt từ file CSV"""
        print("\n📁 CẬP NHẬT HÀNG LOẠT TỪ FILE CSV")
        print("=" * 50)
        print("📋 Format file CSV cần có:")
        print("name,email,phone,age,major,school,position_applied,interview_status,notes")
        print("\n📝 Ví dụ:")
        print("John Doe,john@email.com,0123456789,25,Computer Science,MIT,Software Engineer,scheduled,Good candidate")
        print("Jane Smith,jane@email.com,0987654321,28,Data Science,Stanford,Data Analyst,in-progress,Strong background")
        
        file_path = input("\n📂 Nhập đường dẫn file CSV: ").strip()
        
        if not os.path.exists(file_path):
            print("❌ File không tồn tại!")
            return False
        
        try:
            successful_updates = 0
            failed_updates = 0
            added_new = 0
            updated_existing = 0
            
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                print(f"\n🔄 Bắt đầu xử lý file CSV...")
                
                for row_num, row in enumerate(reader, start=2):
                    try:
                        name = row.get('name', '').strip()
                        if not name:
                            print(f"❌ Dòng {row_num}: Thiếu tên")
                            failed_updates += 1
                            continue
                        
                        # Chuẩn bị dữ liệu
                        data = {}
                        for field in ['email', 'phone', 'major', 'school', 'position_applied', 
                                    'interview_status', 'notes']:
                            value = row.get(field, '').strip()
                            if value and value.lower() != 'n/a':
                                data[field] = value
                        
                        # Xử lý age
                        age_str = row.get('age', '').strip()
                        if age_str and age_str.isdigit():
                            data['age'] = int(age_str)
                        
                        # Kiểm tra người đã tồn tại
                        existing_person = self.db.get_person(name=name)
                        
                        if existing_person:
                            # Cập nhật
                            success = self.db.update_person(person_id=existing_person['id'], **data)
                            if success:
                                print(f"🔄 Cập nhật: {name}")
                                successful_updates += 1
                                updated_existing += 1
                            else:
                                print(f"❌ Không thể cập nhật: {name}")
                                failed_updates += 1
                        else:
                            # Thêm mới
                            person_id = self.db.add_person(name=name, **data)
                            print(f"🆕 Thêm mới: {name} (ID: {person_id})")
                            successful_updates += 1
                            added_new += 1
                            
                    except Exception as e:
                        print(f"❌ Dòng {row_num} ({name}): {e}")
                        failed_updates += 1
            
            print(f"\n📊 KẾT QUẢ IMPORT CSV:")
            print(f"✅ Tổng thành công: {successful_updates}")
            print(f"  🆕 Thêm mới: {added_new}")
            print(f"  🔄 Cập nhật: {updated_existing}")
            print(f"❌ Thất bại: {failed_updates}")
            
            return successful_updates > 0
            
        except Exception as e:
            print(f"❌ Lỗi đọc file: {e}")
            return False

    def export_to_csv(self):
        """Xuất danh sách ứng viên ra CSV"""
        print("\n📤 XUẤT DANH SÁCH ỨNG VIÊN RA CSV")
        print("=" * 50)
        
        try:
            candidates = self.db.list_all_people(active_only=True)
            
            if not candidates:
                print("❌ Không có ứng viên nào để xuất!")
                return False
            
            filename = f"candidates_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = input(f"📂 Tên file (mặc định: {filename}): ").strip() or filename
            
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                # Header
                writer.writerow(['id', 'name', 'email', 'phone', 'age', 'major', 'school', 
                               'position_applied', 'interview_status', 'notes', 'created_at'])
                
                # Data
                for candidate in candidates:
                    writer.writerow([
                        candidate['id'],
                        candidate['name'],
                        candidate.get('email', ''),
                        candidate.get('phone', ''),
                        candidate.get('age', ''),
                        candidate.get('major', ''),
                        candidate.get('school', ''),
                        candidate.get('position_applied', ''),
                        candidate.get('interview_status', ''),
                        candidate.get('notes', ''),
                        candidate.get('created_at', '')
                    ])
            
            print(f"✅ Đã xuất {len(candidates)} ứng viên ra file: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi xuất file: {e}")
            return False

    def create_sample_csv(self):
        """Tạo file CSV mẫu"""
        print("\n📝 TẠO FILE CSV MẪU")
        print("=" * 50)
        
        filename = input("📂 Tên file mẫu (mặc định: sample_candidates.csv): ").strip() or "sample_candidates.csv"
        
        sample_data = [
            ['name', 'email', 'phone', 'age', 'major', 'school', 'position_applied', 'interview_status', 'notes'],
            ['Nguyễn Văn A', 'a@email.com', '0123456789', '25', 'Khoa học máy tính', 'Đại học Bách Khoa', 'Software Engineer', 'scheduled', 'Ứng viên tiềm năng'],
            ['Trần Thị B', 'b@email.com', '0987654321', '28', 'Khoa học dữ liệu', 'Đại học Quốc gia', 'Data Analyst', 'in-progress', 'Kinh nghiệm tốt'],
            ['Lê Văn C', 'c@email.com', '0456789123', '30', 'Quản trị kinh doanh', 'Đại học Kinh tế', 'Product Manager', 'completed', 'Phỏng vấn xuất sắc']
        ]
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(sample_data)
            
            print(f"✅ Đã tạo file CSV mẫu: {filename}")
            print(f"📋 File chứa {len(sample_data)-1} ứng viên mẫu")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi tạo file: {e}")
            return False

    def list_candidates(self):
        """Hiển thị danh sách ứng viên"""
        print("\n👥 DANH SÁCH ỨNG VIÊN")
        print("=" * 50)
        
        candidates = self.db.list_all_people(active_only=True)
        
        if not candidates:
            print("❌ Không có ứng viên nào!")
            return False
        
        print(f"📊 Tìm thấy {len(candidates)} ứng viên:")
        print(f"\n{'STT':<4} {'ID':<4} {'TÊN':<20} {'EMAIL':<25} {'VỊ TRÍ':<20} {'TRẠNG THÁI':<12}")
        print("-" * 90)
        
        for i, candidate in enumerate(candidates, 1):
            name = candidate['name'][:18] + '..' if len(candidate['name']) > 20 else candidate['name']
            email = candidate.get('email') or 'N/A'
            email = email[:23] + '..' if len(email) > 25 else email
            position = candidate.get('position_applied') or 'N/A'
            position = position[:18] + '..' if len(position) > 20 else position
            status = candidate.get('interview_status') or 'N/A'
            status = status[:10] + '..' if len(status) > 12 else status
            
            print(f"{i:<4} {candidate['id']:<4} {name:<20} {email:<25} {position:<20} {status:<12}")
        
        return True

def main():
    """Main function"""
    manager = CandidateManager()
    
    print("👥 QUẢN LÝ ỨNG VIÊN")
    print("=" * 50)
    
    while True:
        print(f"\n🔧 QUẢN LÝ ỨNG VIÊN:")
        print("1. 🆕 Thêm ứng viên mới")
        print("2. 🔄 Cập nhật thông tin ứng viên")
        print("3. 🗑️ Xóa ứng viên")
        print("4. 👥 Xem danh sách ứng viên")
        print("")
        print("📁 QUẢN LÝ CSV:")
        print("5. 📥 Import từ file CSV")
        print("6. 📤 Export ra file CSV")
        print("7. 📝 Tạo file CSV mẫu")
        print("")
        print("0. 🚪 Thoát")
        
        choice = input("\nNhập lựa chọn (0-7): ").strip()
        
        if choice == "0":
            print("👋 Tạm biệt!")
            break
            
        elif choice == "1":
            manager.add_candidate()
            
        elif choice == "2":
            manager.update_candidate()
            
        elif choice == "3":
            manager.delete_candidate()
            
        elif choice == "4":
            manager.list_candidates()
            
        elif choice == "5":
            manager.bulk_import_csv()
            
        elif choice == "6":
            manager.export_to_csv()
            
        elif choice == "7":
            manager.create_sample_csv()
            
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()