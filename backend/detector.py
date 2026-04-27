"""
Egg Crack Detector using YOLOv8n (optimized for CPU)
Includes preprocessing pipeline for LED-backlit mika tray images.
"""

from pathlib import Path
import cv2
import numpy as np

# Try importing ultralytics; graceful fallback if not installed yet
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("WARNING: ultralytics not installed. Run: pip install ultralytics")


MODEL_PATH = Path(__file__).parent / "models" / "best.pt"
FALLBACK_MODEL = "yolov8n.pt"  # base model if custom not trained yet

# Class names matching data.yaml
CLASS_NAMES = {0: "egg_good", 1: "egg_cracked"}

# Colors for bounding boxes (BGR)
COLOR_GOOD = (0, 220, 100)      # Green
COLOR_CRACKED = (0, 60, 255)    # Red
COLOR_TEXT_BG = (20, 20, 20)


class EggDetector:
    def __init__(self, conf_threshold: float = 0.45, iou_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model. Uses custom trained model if available."""
        if not ULTRALYTICS_AVAILABLE:
            print("Ultralytics not available — running in DEMO mode")
            return

        if MODEL_PATH.exists():
            print(f"Loading custom model: {MODEL_PATH}")
            self.model = YOLO(str(MODEL_PATH))
            self.model_loaded = True
            print("Custom egg detection model loaded successfully!")
        else:
            print(f"Custom model not found at {MODEL_PATH}")
            print("Running in DEMO mode — please train the model first.")
            print("See developer panel for instructions.")
            # Load base YOLOv8n for structural demo
            try:
                self.model = YOLO(FALLBACK_MODEL)
                print("Base YOLOv8n loaded (demo mode — not trained for eggs)")
            except Exception as e:
                print(f"Could not load fallback model: {e}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocessing pipeline optimized for LED-backlit mika tray images.
        Enhances crack visibility through contrast and sharpening.
        """
        # 1. Convert to LAB color space for better luminance control
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        #    Boosts local contrast — excellent for revealing cracks under LED light
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # 3. Recombine and convert back to BGR
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 4. Unsharp mask for edge sharpening (makes cracks more visible)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        return sharpened

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run egg detection on a frame.
        Returns detection results + annotated frame.
        """
        if self.model is None:
            # Demo mode: return mock result with original frame
            return self._demo_result(frame)

        # Preprocess for better crack visibility
        processed = self.preprocess(frame)

        # Run YOLOv8 inference (CPU optimized: half=False, device='cpu')
        results = self.model.predict(
            processed,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device="cpu",
            half=False,
            verbose=False,
            imgsz=640,
        )

        # Parse results
        detections = []
        good_count = 0
        cracked_count = 0

        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                label = CLASS_NAMES.get(cls_id, "unknown")
                is_cracked = (cls_id == 1)

                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "cracked": is_cracked,
                })

                if is_cracked:
                    cracked_count += 1
                else:
                    good_count += 1

        # Draw annotations on original frame (not preprocessed)
        annotated = self._draw_annotations(frame.copy(), detections, good_count, cracked_count)

        return {
            "total_eggs": good_count + cracked_count,
            "good_eggs": good_count,
            "cracked_eggs": cracked_count,
            "detections": detections,
            "annotated_frame": annotated,
        }

    def _draw_annotations(self, frame: np.ndarray, detections: list,
                           good: int, cracked: int) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        h, w = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = COLOR_CRACKED if det["cracked"] else COLOR_GOOD
            label = f"{'RETAK' if det['cracked'] else 'BAGUS'} {det['confidence']:.0%}"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Status banner at top
        status = "REJECTED ❌" if cracked > 0 else "ACCEPTED ✅"
        banner_color = (0, 40, 200) if cracked > 0 else (0, 150, 60)
        cv2.rectangle(frame, (0, 0), (w, 36), banner_color, -1)
        cv2.putText(frame, f"{status}  |  Bagus: {good}  Retak: {cracked}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def _demo_result(self, frame: np.ndarray) -> dict:
        """Return demo result when model is not loaded."""
        h, w = frame.shape[:2]
        demo_frame = frame.copy()

        # Draw demo overlay
        cv2.rectangle(demo_frame, (0, 0), (w, 36), (40, 40, 150), -1)
        cv2.putText(demo_frame, "DEMO MODE — Model belum ditraining",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 100), 2)

        return {
            "total_eggs": 0,
            "good_eggs": 0,
            "cracked_eggs": 0,
            "detections": [],
            "annotated_frame": demo_frame,
        }
