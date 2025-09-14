"""
Script liệt kê tất cả tên trong folder data
Tạo danh sách tên từ cấu trúc thư mục để import vào database
"""

import os
import csv
from datetime import datetime

def get_data_folder_path(data_folder="data"):
    """Tìm đường dẫn chính xác đến folder data"""
    # Thử các đường dẫn có thể (folder data nằm trong src)
    possible_paths = [
        data_folder,  # Thư mục hiện tại (nếu chạy từ src)
        os.path.join(".", data_folder),  # ./data (từ src)
        os.path.join("src", data_folder),  # src/data (nếu chạy từ thư mục gốc)
        os.path.join(os.path.dirname(__file__), data_folder),  # Cùng thư mục với script (src/data)
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return os.path.abspath(path)
    
    return None

def list_names_from_data_folder(data_folder="data"):
    """Liệt kê tất cả tên từ folder data"""
    # Tìm đường dẫn chính xác đến folder data
    actual_data_path = get_data_folder_path(data_folder)
    
    if not actual_data_path:
        print(f"❌ Folder không tồn tại: {data_folder}")
        print(f"📁 Thư mục hiện tại: {os.getcwd()}")
        print(f"📁 Script location: {os.path.dirname(__file__)}")
        print(f"📁 Các folder có sẵn: {[f for f in os.listdir('.') if os.path.isdir(f)]}")
        return []
    
    print(f"🔍 Quét folder: {actual_data_path}")
    
    names = []
    for item in os.listdir(actual_data_path):
        item_path = os.path.join(actual_data_path, item)
        if os.path.isdir(item_path):
            names.append(item)
    
    return sorted(names)

def create_csv_from_names(names, output_file="names_from_data.csv"):
    """Tạo file CSV từ danh sách tên"""
    print(f"\n📝 Tạo file CSV: {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['name', 'email', 'phone', 'age', 'major', 'school', 'position_applied', 'interview_status', 'notes'])
        
        # Data rows with empty fields
        for name in names:
            writer.writerow([name, '', '', '', '', '', '', 'scheduled', f'Imported from folder: {name}'])
    
    print(f"✅ Đã tạo file CSV với {len(names)} tên")

def create_detailed_report(names, data_folder="data"):
    """Tạo báo cáo chi tiết về số lượng ảnh trong mỗi folder"""
    # Tìm đường dẫn chính xác đến folder data
    actual_data_path = get_data_folder_path(data_folder)
    if not actual_data_path:
        print(f"❌ Không tìm thấy folder data để tạo báo cáo")
        return 0
        
    print(f"\n📊 BÁO CÁO CHI TIẾT:")
    print("=" * 80)
    print(f"{'STT':<4} {'TÊN':<30} {'SỐ ẢNH':<10} {'ĐỊA CHỈ FOLDER'}")
    print("-" * 80)
    
    total_images = 0
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    for i, name in enumerate(names, 1):
        folder_path = os.path.join(actual_data_path, name)
        
        # Đếm số ảnh trong folder
        image_count = 0
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(supported_formats):
                    image_count += 1
        
        total_images += image_count
        print(f"{i:<4} {name:<30} {image_count:<10} {folder_path}")
    
    print("-" * 80)
    print(f"📈 TỔNG KẾT: {len(names)} người - {total_images} ảnh")
    
    return total_images

def export_to_text_file(names, output_file="names_list.txt"):
    """Export danh sách tên ra file text đơn giản"""
    print(f"\n📄 Tạo file text: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"DANH SÁCH TÊN TRONG FOLDER DATA\n")
        f.write(f"Ngày tạo: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Tổng số: {len(names)} người\n")
        f.write("=" * 50 + "\n\n")
        
        for i, name in enumerate(names, 1):
            f.write(f"{i:3d}. {name}\n")
    
    print(f"✅ Đã tạo file text với {len(names)} tên")

def compare_with_database():
    """So sánh danh sách folder với database hiện tại"""
    try:
        from database_manager import DatabaseManager
        
        db = DatabaseManager("face_recognition.db")
        db_people = db.list_all_people(active_only=True)
        db_names = [person['name'] for person in db_people]
        
        folder_names = list_names_from_data_folder()
        
        print(f"\n🔍 SO SÁNH FOLDER VÀ DATABASE:")
        print("=" * 60)
        print(f"📁 Số người trong folder: {len(folder_names)}")
        print(f"💾 Số người trong database: {len(db_names)}")
        
        # Tìm người có trong folder nhưng không có trong database
        missing_in_db = [name for name in folder_names if name not in db_names]
        if missing_in_db:
            print(f"\n❌ CÓ TRONG FOLDER NHƯNG KHÔNG CÓ TRONG DATABASE ({len(missing_in_db)}):")
            for name in missing_in_db:
                print(f"  - {name}")
        
        # Tìm người có trong database nhưng không có folder
        missing_folder = [name for name in db_names if name not in folder_names]
        if missing_folder:
            print(f"\n📁 CÓ TRONG DATABASE NHƯNG KHÔNG CÓ FOLDER ({len(missing_folder)}):")
            for name in missing_folder:
                print(f"  - {name}")
        
        if not missing_in_db and not missing_folder:
            print(f"\n✅ ĐỒNG BỘ HOÀN HẢO! Folder và database khớp nhau.")
        
        return missing_in_db, missing_folder
        
    except ImportError:
        print("❌ Không thể import DatabaseManager")
        return [], []
    except Exception as e:
        print(f"❌ Lỗi khi so sánh với database: {e}")
        return [], []

def main():
    """Main function"""
    print("📁 LIỆT KÊ TÊN TRONG FOLDER DATA")
    print("=" * 50)
    
    # Lấy danh sách tên
    names = list_names_from_data_folder()
    
    if not names:
        print("❌ Không tìm thấy folder nào trong data!")
        return
    
    # Hiển thị danh sách
    print(f"\n👥 Tìm thấy {len(names)} người:")
    for i, name in enumerate(names, 1):
        print(f"  {i:2d}. {name}")
    
    # Tạo báo cáo chi tiết
    total_images = create_detailed_report(names)
    
    # So sánh với database
    missing_in_db, missing_folder = compare_with_database()
    
    # Menu tùy chọn
    while True:
        print(f"\n🔧 TÙY CHỌN:")
        print("1. 📝 Tạo file CSV (để import vào database)")
        print("2. 📄 Tạo file text đơn giản")
        print("3. 📊 Hiển thị lại báo cáo chi tiết")
        print("4. 🔍 So sánh lại với database")
        print("5. 📁 Quét lại folder data")
        print("0. 🚪 Thoát")
        
        choice = input("\nNhập lựa chọn (0-5): ").strip()
        
        if choice == "0":
            print("👋 Tạm biệt!")
            break
            
        elif choice == "1":
            output_csv = input("Tên file CSV (mặc định: names_from_data.csv): ").strip() or "names_from_data.csv"
            create_csv_from_names(names, output_csv)
            print(f"💡 Để import vào database: python candidate_manager.py → chọn 5 → nhập '{output_csv}'")
            
        elif choice == "2":
            output_txt = input("Tên file text (mặc định: names_list.txt): ").strip() or "names_list.txt"
            export_to_text_file(names, output_txt)
            
        elif choice == "3":
            create_detailed_report(names)
            
        elif choice == "4":
            missing_in_db, missing_folder = compare_with_database()
            
        elif choice == "5":
            names = list_names_from_data_folder()
            print(f"🔄 Đã quét lại. Tìm thấy {len(names)} người.")
            
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()