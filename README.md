# 🔍 Phát hiện Giao dịch Gian lận

Web app Streamlit để huấn luyện, so sánh mô hình ML và dự báo giao dịch gian lận (fraud detection).

---

## 📁 Cấu trúc thư mục

```
├── app.py               # Ứng dụng Streamlit chính
├── requirements.txt     # Danh sách thư viện cần cài
├── README.md            # Tài liệu hướng dẫn (file này)
└── dataset1.csv         # Dữ liệu mẫu để huấn luyện
```

---

## ⚙️ Cài đặt & Chạy

### 1. Clone / tải project về máy

```bash
git clone <your-repo-url>
cd fraud-detection-app
```

### 2. (Tuỳ chọn) Tạo môi trường ảo

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Cài thư viện

```bash
pip install -r requirements.txt
```

### 4. Chạy ứng dụng

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`

---

## 🗂️ Định dạng dữ liệu

### File huấn luyện (CSV)

| Cột | Mô tả |
|-----|-------|
| `X_1` … `X_14` | 14 đặc trưng số (float) |
| `default` | Nhãn: `0` = bình thường, `1` = gian lận |

Ví dụ:

```
X_1,X_2,...,X_14,default
0.026,0.191,...,0.841,0
0.043,0.045,...,0.871,1
```

### File dự báo hàng loạt (CSV)

Chỉ cần 14 cột đặc trưng `X_1` … `X_14`, **không cần** cột `default`.

---

## 📊 Tính năng

### Tab 1 — So sánh mô hình
- Bảng tổng hợp Accuracy, Precision, Recall, F1, AUC-ROC cho 3 mô hình
- Tô màu xanh ô có giá trị tốt nhất
- Biểu đồ bar chart so sánh trực quan
- Thống kê phân bố nhãn dữ liệu

### Tab 2 — Chi tiết & ROC
- Ma trận nhầm lẫn (Confusion Matrix)
- Đường cong ROC của cả 3 mô hình trên cùng biểu đồ
- Báo cáo phân loại chi tiết (classification report)
- Biểu đồ Feature Importance (Random Forest / Decision Tree)

### Tab 3 — Dự báo đơn lẻ
- Nhập thủ công 14 giá trị đặc trưng
- Kết quả: nhãn dự báo + xác suất gian lận
- Biểu đồ thanh thể hiện xác suất

### Tab 4 — Dự báo hàng loạt
- Upload file CSV nhiều giao dịch
- Hiển thị kết quả với hàng gian lận tô màu đỏ
- Tải xuống file kết quả CSV

---

## 🤖 Mô hình ML

| Mô hình | Ghi chú |
|---------|---------|
| Logistic Regression | Nền tảng, nhanh, dễ giải thích |
| Decision Tree | Trực quan, dễ visualize |
| Random Forest | Thường cho kết quả tốt nhất |

Tất cả mô hình được huấn luyện với tham số `random_state` và `test_size` tuỳ chỉnh qua sidebar.

---

## 🛠️ Yêu cầu hệ thống

- Python >= 3.9
- Các thư viện: xem `requirements.txt`

---

## 📝 Ghi chú

- Dữ liệu mẫu: 1 386 giao dịch, trong đó 151 gian lận (~10.9%)
- Mô hình được cache lại sau lần huấn luyện đầu tiên, thay đổi tham số sẽ tự động huấn luyện lại
- Kết quả dự báo hàng loạt có thể tải về dạng CSV (UTF-8 BOM, mở được bằng Excel)
