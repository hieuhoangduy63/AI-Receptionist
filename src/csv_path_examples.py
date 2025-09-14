"""
VÍ DỤ CHI TIẾT: CÁCH NHẬP ĐƯỜNG DẪN FILE CSV KHI SỬ DỤNG CANDIDATE_MANAGER
"""

print("🎯 VÍ DỤ CÁC CÁCH NHẬP ĐƯỜNG DẪN FILE CSV")
print("=" * 60)

print("\n📂 CẤU TRÚC THỦ MỤC HIỆN TẠI:")
print("E:/face recognition/TRAE_AI_Receptionist/")
print("├── src/")
print("│   ├── candidate_manager.py")
print("│   ├── addCandidate0           ← File CSV của bạn")
print("│   ├── candidates_sample.csv")
print("│   └── other_files.py")
print("├── data/")
print("└── models/")

print("\n🔧 KHI CHẠY: python candidate_manager.py")
print("   Chọn: 5. 📥 Import từ file CSV")
print("   Hệ thống sẽ hỏi: 'Nhập đường dẫn file CSV:'")

print("\n📝 CÁC CÁCH NHẬP ĐƯỜNG DẪN:")

print("\n1️⃣ ĐƯỜNG DẪN TƯƠNG ĐỐI (từ thư mục src/):")
print("   ✅ addCandidate0")
print("   ✅ ./addCandidate0") 
print("   ✅ candidates_sample.csv")

print("\n2️⃣ ĐƯỜNG DẪN TUYỆT ĐỐI:")
print("   ✅ E:/face recognition/TRAE_AI_Receptionist/src/addCandidate0")
print("   ✅ e:\\face recognition\\TRAE_AI_Receptionist\\src\\addCandidate0")

print("\n3️⃣ FILE Ở THỦ MỤC KHÁC:")
print("   ✅ ../data/some_file.csv                    (thư mục data)")
print("   ✅ C:/Users/Desktop/my_candidates.csv       (Desktop)")
print("   ✅ D:/Documents/candidates.csv              (Ổ D)")

print("\n4️⃣ FILE CÓ KHOẢNG TRẮNG TRONG TÊN:")
print('   ✅ "My Candidates File.csv"')
print('   ✅ "C:/My Documents/Candidate List.csv"')

print("\n🎯 VÍ DỤ CỤ THỂ CHO FILE addCandidate0 CỦA BẠN:")
print("=" * 60)

print("\n🔥 CÁCH ĐƠN GIẢN NHẤT (Khuyến nghị):")
print("   📂 Nhập đường dẫn file CSV: addCandidate0")
print("   💡 Vì file đã ở trong thư mục src/")

print("\n🔥 CÁCH ĐẦY ĐỦ:")
print("   📂 Nhập đường dẫn file CSV: E:/face recognition/TRAE_AI_Receptionist/src/addCandidate0")

print("\n⚡ DEMO THỰC TẾ:")
print("=" * 60)
print("🔧 Bước 1: cd 'e:/face recognition/TRAE_AI_Receptionist/src'")
print("🔧 Bước 2: python candidate_manager.py")
print("🔧 Bước 3: Chọn '5'")
print("🔧 Bước 4: Nhập 'addCandidate0'")
print("🔧 Bước 5: Xem kết quả import!")

print("\n📊 KẾT QUẢ MONG ĐỢI:")
print("✅ Thành công: 16 ứng viên (từ Ali Naimi đến Vladimir Putin)")
print("❌ Thất bại: 0 ứng viên")

print("\n🚨 LƯU Ý QUAN TRỌNG:")
print("• File CSV phải có encoding UTF-8")
print("• Delimiter phải là dấu phẩy (,)")
print("• Header phải đúng: name,email,phone,age,major,school,position_applied,interview_status,notes")
print("• Trường 'name' là bắt buộc, các trường khác tùy chọn")
print("• Nếu ứng viên đã tồn tại, hệ thống sẽ cập nhật thông tin")

print("\n🛠️ KHẮC PHỤC SỰ CỐ:")
print("❌ File not found → Kiểm tra đường dẫn và tên file")  
print("❌ Encoding error → Lưu file với UTF-8 encoding")
print("❌ Header error → Kiểm tra tên cột trong dòng đầu tiên")
print("❌ Data error → Kiểm tra dữ liệu không có ký tự đặc biệt")

print("\n🎯 SẴN SÀNG TEST:")
print("File addCandidate0 của bạn đã được sửa header và sẵn sàng import!")
print("Chạy ngay: python candidate_manager.py → chọn 5 → nhập 'addCandidate0'")