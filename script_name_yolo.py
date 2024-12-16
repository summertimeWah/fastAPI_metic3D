from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from supabase import create_client, Client
import numpy as np
import cv2
import os
import base64
import json
import random
import logging
from datetime import datetime
import torch
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from models.experimental import attempt_load

# 初始化 FastAPI 應用程式
app = FastAPI()

# 這些變數控制著應用程序的基本運行
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)

frame_count = 0
dangerLevel = 0
width = 0
height = 0

# 設置 Supabase Client
SUPABASE_URL = "https://oxskmydkkwzllyxnbcny.supabase.co"  # 替換為您的 Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im94c2tteWRra3d6bGx5eG5iY255Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyNzg0MjEyMywiZXhwIjoyMDQzNDE4MTIzfQ.tDqV4zXnhChIlDN0EUHJaPSogFjtIWTBLDxuufX1hDs"  # 替換為您的 API 金鑰
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

response = supabase.table("video").select("*").execute()

user_id = "61666aaa-2d3a-4898-90fa-1d23bda31fc2"

# YOLO 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = 'yolov7.pt'  # 替換為你的模型路徑
model = attempt_load(weights, map_location=device)
model.eval()

# WebSocket 端點
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
            width = message.get('width')
            height = message.get('height')

            if base64_image and width and height:
                frame_data = base64.b64decode(base64_image)
                frame = process_nv21_frame(frame_data, width, height)

                # 使用 YOLO 模型進行檢測
                detections = yolo_detect(frame)

                # 保存處理後的圖片
                frame_path = os.path.join(FRAME_DIR, f"frame_{frame_count:05d}.png")
                cv2.imwrite(frame_path, frame)
                frame_count += 1

                # 假設 danger level 取決於偵測結果
                danger_level = random.randint(1, 4) if detections else 0

                await websocket.send_text(json.dumps({"status": "frame received", "danger": danger_level, "detections": detections}))
    except WebSocketDisconnect:
        logger.debug("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        await finalize_video(logger)
        logger.debug("Video creation finalized")

# 處理 NV21 格式的影像
def process_nv21_frame(nv21_data, width, height):
    expected_size = width * height * 3 // 2
    if len(nv21_data) != expected_size:
        raise ValueError(f"Invalid frame size: expected {expected_size}, got {len(nv21_data)}")

    yuv_image = np.frombuffer(nv21_data, np.uint8).reshape((height * 3 // 2, width))
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    
    # 顯示影像以驗證處理
    cv2.imshow("Frame", bgr_image)
    cv2.waitKey(1)

    return bgr_image

# 使用 YOLO 模型進行檢測
def yolo_detect(frame):
    # 將影像縮放至模型要求的大小
    img = letterbox(frame, 640, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # 準備張量並傳入模型
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # 歸一化
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # YOLO 偵測
    with torch.no_grad():
        pred = model(img)[0]

    # 使用 NMS 去除多餘的框
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
# 最後將所有幀合併成影片並上傳到 Supabase
async def finalize_video(logger):
    global frame_count, width, height

    # 保存影片
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (width, height))

    for i in range(frame_count):
        frame_path = os.path.join(FRAME_DIR, f"frame_{i:05d}.png")
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            video_writer.write(frame)

    video_writer.release()
    cv2.destroyAllWindows()

    # 清理臨時圖片
    for i in range(frame_count):
        os.remove(os.path.join(FRAME_DIR, f"frame_{i:05d}.png"))

    frame_count = 0
    output_video_path = "output_video.mp4"

    try:
        # 將影片上傳到 Supabase
        bucket_name = "video"
        video_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # 讀取影片檔案並上傳
        with open(output_video_path, 'rb') as video_file:
            video_data = video_file.read()

        response = supabase.storage.from_("video").upload(
            file=video_data,
            path=video_name,
            file_options={"content-type": "video/mp4"}
        )

        # 檢查是否上傳成功
        if response.status_code != 200:
            logger.error(f"Error uploading video to Supabase: {response.json()}")
            return
        
        # 生成影片 URL 並插入資料表
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
