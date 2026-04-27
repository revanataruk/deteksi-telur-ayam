import asyncio
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from detector import EggDetector

app = FastAPI(title="Egg Crack Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector (loads model on startup)
detector = EggDetector()

# Session stats (in-memory)
stats = {
    "mika_count": 0,
    "total_eggs": 0,
    "good_eggs": 0,
    "cracked_eggs": 0,
    "accepted_mika": 0,
    "rejected_mika": 0,
}


@app.get("/")
async def root():
    return {"status": "Egg Detector API Running", "model_loaded": detector.model_loaded}


@app.get("/stats")
async def get_stats():
    return stats


@app.post("/reset-stats")
async def reset_stats():
    for key in stats:
        stats[key] = 0
    return {"message": "Stats reset successfully"}


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time egg detection.
    Client sends: base64-encoded JPEG frame
    Server returns: detection results + annotated frame
    """
    await websocket.accept()
    print("WebSocket client connected")

    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            payload = json.loads(data)

            frame_b64 = payload.get("frame")
            if not frame_b64:
                continue

            # Decode base64 to OpenCV image
            img_bytes = base64.b64decode(frame_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            start_time = time.time()

            # Run detection
            result = detector.detect(frame)

            inference_time = round((time.time() - start_time) * 1000, 1)

            # Update stats if detection triggered (e.g., via button press)
            if payload.get("count_this"):
                stats["mika_count"] += 1
                stats["total_eggs"] += result["total_eggs"]
                stats["good_eggs"] += result["good_eggs"]
                stats["cracked_eggs"] += result["cracked_eggs"]
                if result["cracked_eggs"] > 0:
                    stats["rejected_mika"] += 1
                else:
                    stats["accepted_mika"] += 1

            # Encode annotated frame back to base64
            _, buffer = cv2.imencode(".jpg", result["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

            # Send result
            response = {
                "status": "rejected" if result["cracked_eggs"] > 0 else "accepted",
                "total_eggs": result["total_eggs"],
                "good_eggs": result["good_eggs"],
                "cracked_eggs": result["cracked_eggs"],
                "detections": result["detections"],
                "annotated_frame": annotated_b64,
                "inference_ms": inference_time,
                "stats": stats,
                "model_loaded": detector.model_loaded,
            }

            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except:
            pass
