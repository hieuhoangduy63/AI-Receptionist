#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script khởi động AI Receptionist Server
"""

import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("🤖 AI Receptionist System with HR Management")
    print("=" * 60)
    print("📁 Database: src/face_recognition.db")
    print("🧠 Model: models/arcface.onnx")
    print("🌐 Server sẽ chạy tại:")
    print("   👉 http://localhost:8000 (Click để mở trang chủ)")
    print("   👉 http://localhost:8000/ai_receptionist (AI Receptionist)")
    print("   👉 http://localhost:8000/docs (API Documentation)")
    print("=" * 60)
    print("🔑 Default HR Login:")
    print("   Username: admin")
    print("   Password: admin123")
    print("=" * 60)
    print("🚀 Đang khởi động server...")
    print("   (Nhấn Ctrl+C để dừng server)")
    print("=" * 60)
    
    # Khởi động uvicorn server
    try:
        subprocess.run([
            "uvicorn", 
            "src.api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server đã được dừng.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khởi động server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()