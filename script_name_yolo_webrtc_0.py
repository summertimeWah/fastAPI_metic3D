from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from supabase import create_client, Client
import numpy as np
import cv2
import os
import base64
import json
import logging
from datetime import datetime
import torch
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from models.experimental import attempt_load
import asyncio

# Initialize FastAPI
app = FastAPI()

# Setup directories and constants
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)

frame_count = 0
dangerLevel = 0
width, height = 0, 0

# Supabase Client Setup
SUPABASE_URL = "https://oxskmydkkwzllyxnbcny.supabase.co"  # 替換為您的 Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im94c2tteWRra3d6bGx5eG5iY255Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyNzg0MjEyMywiZXhwIjoyMDQzNDE4MTIzfQ.tDqV4zXnhChIlDN0EUHJaPSogFjtIWTBLDxuufX1hDs"  # 替換為您的 API 金鑰supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
user_id = "61666aaa-2d3a-4898-90fa-1d23bda31fc2"

# YOLO Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = 'yolov7.pt'
model = attempt_load(weights, map_location=device)
model.eval()

# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global frame_count, width, height
    await websocket.accept()
    
    logger = logging.getLogger('uvicorn.error')
    logger.setLevel(logging.DEBUG)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            base64_image = message.get('frame')
            width, height = message.get('width'), message.get('height')
            
            if base64_image and width and height:
                frame_data = base64.b64decode(base64_image)
                frame = process_nv21_frame(frame_data, width, height)

                # YOLO Detection on processed frame
                detections = yolo_detect(frame)

                # Save the processed frame
                frame_path = os.path.join(FRAME_DIR, f"frame_{frame_count:05d}.png")
                cv2.imwrite(frame_path, frame)
                frame_count += 1

                # Calculate danger level based on detection results
                danger_level = random.randint(1, 4) if detections else 0

                # Send status and detection results back to the client
                await websocket.send_text(json.dumps({"status": "frame received", "danger": danger_level, "detections": detections}))
    except WebSocketDisconnect:
        logger.debug("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        await finalize_video(logger)
        logger.debug("Video creation finalized")

# Frame processing for NV21 format
def process_nv21_frame(nv21_data, width, height):
    expected_size = width * height * 3 // 2
    if len(nv21_data) != expected_size:
        raise ValueError(f"Invalid frame size: expected {expected_size}, got {len(nv21_data)}")

    yuv_image = np.frombuffer(nv21_data, np.uint8).reshape((height * 3 // 2, width))
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    
    cv2.imshow("Frame", bgr_image)
    cv2.waitKey(1)

    return bgr_image

# YOLO detection
def yolo_detect(frame):
    img = letterbox(frame, 640, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(pred, 0.25, 0.45)

    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                detections.append({
                    "bbox": [int(x) for x in xyxy],
                    "confidence": float(conf),
                    "class": int(cls)
                })
    
    return detections

# Video finalization and upload
async def finalize_video(logger):
    global frame_count, width, height

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (width, height))

    for i in range(frame_count):
        frame_path = os.path.join(FRAME_DIR, f"frame_{i:05d}.png")
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            video_writer.write(frame)

    video_writer.release()
    cv2.destroyAllWindows()

    for i in range(frame_count):
        os.remove(os.path.join(FRAME_DIR, f"frame_{i:05d}.png"))

    frame_count = 0
    output_video_path = "output_video.mp4"

    try:
        bucket_name = "video"
        video_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        with open(output_video_path, 'rb') as video_file:
            video_data = video_file.read()

        response = supabase.storage.from_("video").upload(
            file=video_data,
            path=video_name,
            file_options={"content-type": "video/mp4"}
        )

        if response.status_code != 200:
            logger.error(f"Error uploading video to Supabase: {response.json()}")
            return
        
        video_url = f"https://{SUPABASE_URL.replace('https://', '')}/storage/v1/object/public/{bucket_name}/{video_name}"

        video_record = {
            "uid": user_id,
            "url": video_url,
            "videoName": video_name,
            "created_at": datetime.utcnow().isoformat()
        }

        insert_response = supabase.table('video').insert(video_record).execute()

        if insert_response.get('error'):
            logger.error(f"Error inserting video record into Supabase: {insert_response.json()}")
        else:
            logger.debug("Video record inserted successfully.")

    except Exception as e:
        logger.error(f"Error processing video upload: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
