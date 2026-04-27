# 🥚 EggGuard — Sistem Deteksi Telur Retak

Sistem deteksi telur retak berbasis AI (YOLOv8) untuk home industri telur ayam.
Menggunakan kamera HP (top-down) + LED panel sebagai backlight untuk mendeteksi retakan.

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Backend
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Buka Frontend
- **User (produksi):** Buka `frontend/user.html` di browser
- **Developer (setup):** Buka `frontend/developer.html` di browser

---

## 📁 Struktur Project

```
egg-detector/
├── backend/
│   ├── main.py          ← FastAPI server + WebSocket
│   ├── detector.py      ← YOLOv8 inference + preprocessing
│   └── models/
│       └── best.pt      ← (diisi setelah training)
├── frontend/
│   ├── user.html        ← Halaman deteksi produksi
│   └── developer.html   ← Panduan training & setup
├── dataset/
│   ├── images/train/    ← Foto untuk training
│   ├── images/val/      ← Foto untuk validasi
│   ├── labels/train/    ← Anotasi .txt (YOLO format)
│   └── labels/val/
├── train.py             ← Script training YOLOv8n
├── data.yaml            ← Config dataset
└── requirements.txt
```

---

## 🔄 Alur Kerja

```
HP (IP Webcam App)
    ↓ MJPEG stream
Browser (user.html)
    ↓ frame base64 via WebSocket
FastAPI Backend
    ↓ CLAHE + Unsharp masking (OpenCV)
    ↓ YOLOv8n inference
    ↓ Result: {good, cracked, positions}
Browser
    ├── Video + bounding boxes
    ├── Status: ACCEPTED ✅ / REJECTED ❌
    ├── Audio feedback (beep berbeda)
    └── Counter session stats
```

---

## 📸 Setup Fisik

```
[HP dengan IP Webcam]
        | (penyangga, kamera menghadap bawah)
        ↓ top-down view
[Mika Telur isi 10]
[LED Panel (seukuran mika)]
```

1. **LED Panel** di paling bawah → menyinari telur dari bawah
2. **Mika** diletakkan di atas LED panel
3. **HP** di atas dengan kamera menghadap ke bawah (sudut ~90°)
4. Semua terhubung ke laptop via WiFi (HP) dan USB/WiFi (laptop)

---

## 🏋️ Training Model

### Langkah Singkat:
1. Kumpulkan foto mika (min 100 foto, berbagai skenario)
2. Anotasi dengan LabelImg: `pip install labelImg && labelImg`
3. Simpan ke `dataset/images/` dan `dataset/labels/`
4. Jalankan: `python train.py`
5. Model otomatis disalin ke `backend/models/best.pt`
6. Restart backend

### Target Akurasi:
- mAP50 > 0.85 = siap produksi
- Lihat grafik di `runs/egg_detector/results.png`

---

## 🎛️ Konfigurasi

### Confidence Threshold (`backend/detector.py`)
```python
conf_threshold = 0.45  # Naikkan jika false positive banyak
```

### FPS (`frontend/user.html`)
```javascript
const FPS_TARGET = 8;  # Kurangi jika CPU berat
```

### IP Webcam URL
Format: `http://[IP_HP]:8080/video`
Ganti di halaman user atau di `frontend/user.html`

---

## 🔊 Audio Feedback

| Kondisi | Suara |
|---------|-------|
| ACCEPTED (semua bagus) | 2 beep tinggi naik (880Hz → 1046Hz) |
| REJECTED (ada retak) | 2 buzz rendah (220Hz, sawtooth) |

Audio dimainkan otomatis saat tombol **"Hitung & Catat Mika Ini"** ditekan.

---

## 📊 Counter & Statistik

Per sesi (reset tiap restart atau tekan Reset):
- Jumlah mika diproses
- Total telur (semua mika)
- Total telur bagus
- Total telur retak
- Mika diterima (ACCEPTED)
- Mika ditolak (REJECTED)

---

## ❓ Troubleshooting

**Backend tidak bisa diakses:**
```bash
# Pastikan port 8000 tidak dipakai
uvicorn main:app --host 0.0.0.0 --port 8000
```

**HP tidak terdeteksi:**
- Pastikan HP dan laptop di WiFi yang sama
- Coba ping IP HP dari laptop
- Cek firewall laptop (allow port 8080)

**Deteksi tidak akurat (demo mode):**
- Training belum dilakukan → ikuti panduan di developer.html
- Model belum disalin ke `backend/models/best.pt`

**CPU terlalu berat (lag):**
- Kurangi `FPS_TARGET` di user.html (misal ke 5)
- Kurangi resolusi kamera di IP Webcam app
