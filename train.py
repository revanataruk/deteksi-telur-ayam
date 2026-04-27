"""
Training script untuk model deteksi telur retak.
Jalankan setelah dataset sudah dianotasi lengkap.

Cara pakai:
    python train.py

Pastikan dataset sudah ada di folder dataset/ dengan struktur:
    dataset/
    ├── images/train/   (foto training)
    ├── images/val/     (foto validasi)
    ├── labels/train/   (anotasi .txt YOLOv8 format)
    └── labels/val/
"""

from pathlib import Path
import shutil
from ultralytics import YOLO

# ── CONFIG ──────────────────────────────────────────────────────────────
DATASET_YAML = Path(__file__).parent / "data.yaml"
OUTPUT_DIR   = Path(__file__).parent / "runs"
MODEL_DST    = Path(__file__).parent / "backend" / "models" / "best.pt"

EPOCHS      = 100       # Tambah ke 150-200 kalau dataset besar (>500 foto)
IMGSZ       = 640       # Resolusi training
BATCH       = 8         # Turunkan ke 4 kalau RAM < 8GB
PATIENCE    = 20        # Early stopping — stop jika tidak ada peningkatan
WORKERS     = 2         # CPU workers untuk data loading
DEVICE      = "cpu"     # Ganti ke "0" kalau ada GPU NVIDIA
# ────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  EGG CRACK DETECTOR — YOLOv8n Training")
    print("=" * 60)

    # Cek dataset
    if not DATASET_YAML.exists():
        print(f"ERROR: data.yaml tidak ditemukan di {DATASET_YAML}")
        return

    train_img = Path("dataset/images/train")
    val_img   = Path("dataset/images/val")
    if not train_img.exists() or not any(train_img.iterdir()):
        print("ERROR: Folder dataset/images/train kosong atau tidak ada.")
        print("Silakan lihat panduan di halaman Developer untuk mengumpulkan foto.")
        return

    n_train = len(list(train_img.glob("*.jpg")) + list(train_img.glob("*.png")))
    n_val   = len(list(val_img.glob("*.jpg")) + list(val_img.glob("*.png"))) if val_img.exists() else 0
    print(f"Dataset: {n_train} foto training, {n_val} foto validasi")

    if n_train < 20:
        print("WARNING: Dataset sangat kecil (<20 foto). Akurasi mungkin rendah.")
        print("Disarankan minimal 100 foto training untuk hasil yang baik.")

    # Load YOLOv8 nano — terkecil dan tercepat, ideal untuk CPU
    print("\nLoading YOLOv8n base model...")
    model = YOLO("yolov8n.pt")

    print(f"\nMemulai training selama {EPOCHS} epoch...")
    print("(Ini bisa memakan waktu 30-90 menit di CPU)\n")

    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        workers=WORKERS,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name="egg_detector",
        exist_ok=True,
        # Augmentation — sangat penting untuk dataset kecil
        hsv_h=0.015,    # Warna (minimal, karena LED bisa beda)
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=5.0,    # Rotasi kecil (mika biasanya lurus)
        translate=0.1,
        scale=0.3,
        flipud=0.3,     # Flip vertikal (posisi telur bisa beda)
        fliplr=0.5,     # Flip horizontal
        mosaic=0.5,     # Mosaic augmentation
        # Optimizer settings untuk CPU
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Output
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Training selesai!")

    # Copy best model ke folder backend
    best_model_src = OUTPUT_DIR / "egg_detector" / "weights" / "best.pt"
    if best_model_src.exists():
        MODEL_DST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model_src, MODEL_DST)
        print(f"✅ Model terbaik disalin ke: {MODEL_DST}")
        print("   Sekarang restart backend untuk menggunakan model baru.")
    else:
        print(f"WARNING: best.pt tidak ditemukan di {best_model_src}")

    print("=" * 60)


if __name__ == "__main__":
    main()
