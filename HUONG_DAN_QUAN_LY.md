# 📋 HƯỚNG DẪN CHI TIẾT: QUẢN LÝ ỨNG VIÊN VÀ LỊCH HẸN

## 🎯 TỔNG QUAN HỆ THỐNG

Hệ thống được chia thành **2 phần chính**:

### 📁 **1. QUẢN LÝ ỨNG VIÊN** (`candidate_manager.py`)

- ✅ **Thêm/Sửa/Xóa** ứng viên
- ✅ **Import hàng loạt** từ file CSV
- ✅ **Export** danh sách ra CSV

### 📅 **2. QUẢN LÝ LỊCH HẸN** (`interview_manager.py`)

- ✅ **Tạo/Sửa/Xóa** lịch phỏng vấn
- ✅ **Quản lý phòng họp** và thời gian
- ✅ **Theo dõi trạng thái** phỏng vấn

---

## 📝 CẬP NHẬT HÀNG LOẠT TỪ FILE CSV

### **🔧 Cách 1: Sử dụng Script**

```bash
cd "e:/face recognition/TRAE_AI_Receptionist/src"
python candidate_manager.py
```

**Chọn tùy chọn 5**: 📥 Import từ file CSV

### **📊 Format file CSV chuẩn:**

```csv
name,email,phone,age,major,school,position_applied,interview_status,notes
Nguyễn Văn A,a@email.com,0901234567,25,Khoa học máy tính,Đại học Bách Khoa,Software Engineer,scheduled,Ứng viên tiềm năng
Trần Thị B,b@email.com,0912345678,27,Quản trị kinh doanh,Đại học Kinh tế,Business Analyst,in-progress,Có kinh nghiệm 3 năm
```

### **📋 Mô tả các trường:**

| Trường             | Bắt buộc  | Mô tả                   | Ví dụ                |
| ------------------ | --------- | ----------------------- | -------------------- |
| `name`             | ✅ **Có** | Tên ứng viên (duy nhất) | `Nguyễn Văn A`       |
| `email`            | ❌ Không  | Địa chỉ email           | `a@email.com`        |
| `phone`            | ❌ Không  | Số điện thoại           | `0901234567`         |
| `age`              | ❌ Không  | Tuổi (số nguyên)        | `25`                 |
| `major`            | ❌ Không  | Chuyên ngành            | `Khoa học máy tính`  |
| `school`           | ❌ Không  | Trường đại học          | `Đại học Bách Khoa`  |
| `position_applied` | ❌ Không  | Vị trí ứng tuyển        | `Software Engineer`  |
| `interview_status` | ❌ Không  | Trạng thái phỏng vấn    | `scheduled`          |
| `notes`            | ❌ Không  | Ghi chú                 | `Ứng viên tiềm năng` |

### **📌 Các trạng thái phỏng vấn hợp lệ:**

- `scheduled` - Đã lên lịch
- `in-progress` - Đang tiến hành
- `completed` - Hoàn thành
- `hired` - Đã tuyển dụng
- `rejected` - Từ chối

---

## 🚀 HƯỚNG DẪN THỰC HÀNH

### **📥 Bước 1: Chuẩn bị file CSV**

1. **Tạo file CSV mẫu:**

   ```bash
   python candidate_manager.py
   # Chọn "7. 📝 Tạo file CSV mẫu"
   ```

2. **Chỉnh sửa file CSV** với dữ liệu thực tế

3. **Kiểm tra định dạng:**
   - Encoding: UTF-8
   - Delimiter: dấu phẩy (,)
   - Header: đúng tên cột

### **📥 Bước 2: Import dữ liệu**

1. **Chạy script:**

   ```bash
   python candidate_manager.py
   ```

2. **Chọn "5. 📥 Import từ file CSV"**

3. **Nhập đường dẫn file** (ví dụ: `candidates_sample.csv`)

4. **Xem kết quả import:**
   - ✅ Thành công: X ứng viên
   - ❌ Thất bại: Y ứng viên (với lý do)

### **📤 Bước 3: Export và kiểm tra**

1. **Export dữ liệu hiện tại:**

   ```bash
   # Chọn "6. 📤 Export ra file CSV"
   ```

2. **Kiểm tra danh sách:**
   ```bash
   python check_candidates.py
   ```

---

## ⚙️ TÍNH NĂNG XÓA ỨNG VIÊN

### **🗑️ Soft Delete (Khuyến nghị)**

- Ứng viên **không bị xóa khỏi database**
- Chỉ đặt `active = FALSE`
- **Có thể khôi phục** sau này
- **Giữ nguyên dữ liệu** face embeddings

### **🔥 Hard Delete (Cẩn thận)**

- Ứng viên **bị xóa hoàn toàn** khỏi database
- **Không thể khôi phục**
- **Mất tất cả dữ liệu** liên quan

### **📝 Cách xóa:**

```bash
python candidate_manager.py
# Chọn "3. 🗑️ Xóa ứng viên"
# Tìm theo ID hoặc tên
# Xác nhận bằng cách gõ "XAC NHAN"
```

---

## 📅 QUẢN LÝ LỊCH HẸN PHỎNG VẤN

### **🔧 Chạy script quản lý lịch:**

```bash
python interview_manager.py
```

### **🎯 Các tính năng:**

1. **📅 Tạo lịch hẹn mới**
2. **🔄 Cập nhật lịch hẹn**
3. **🗑️ Hủy lịch hẹn**
4. **👥 Xem lịch theo ngày/tuần**
5. **🏢 Quản lý phòng họp**

### **📋 Thông tin lịch hẹn:**

- **Ứng viên** (liên kết với ID)
- **HR phụ trách**
- **Ngày giờ** phỏng vấn
- **Phòng họp**
- **Loại phỏng vấn** (technical/hr/final)
- **Thời lượng** (phút)
- **Trạng thái** và ghi chú

---

## 🔧 KHẮC PHỤC SỰ CỐ

### **❌ Lỗi "Không thể xóa ứng viên"**

- **Nguyên nhân:** Trường `active` chưa được hỗ trợ
- **Giải pháp:** Đã sửa trong `database_manager.py`

### **❌ Lỗi import CSV**

- **Kiểm tra encoding:** UTF-8
- **Kiểm tra delimiter:** dấu phẩy
- **Kiểm tra header:** đúng tên cột
- **Kiểm tra dữ liệu:** không có ký tự đặc biệt

### **❌ Ứng viên trùng tên**

- **Script sẽ cập nhật** thông tin ứng viên hiện có
- **Không tạo duplicate**

---

## 📊 FILE DEMO CÓ SẴN

1. **`candidates_sample.csv`** - 8 ứng viên mẫu
2. **`check_candidates.py`** - Xem thông tin ứng viên
3. **`candidate_manager.py`** - Quản lý ứng viên
4. **`interview_manager.py`** - Quản lý lịch hẹn

**🚀 Bắt đầu ngay:**

```bash
cd "e:/face recognition/TRAE_AI_Receptionist/src"
python candidate_manager.py
# Chọn "5" và nhập "candidates_sample.csv"
```
