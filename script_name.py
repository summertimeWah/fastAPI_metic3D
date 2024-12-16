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

# 初始化 FastAPI 應用程式
app = FastAPI()

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

# 打印 response 的類型和內容
print(type(response))  # 打印 response 的類型
print(response)  # 打印 response 的完整內容


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

                frame_path = os.path.join(FRAME_DIR, f"frame_{frame_count:05d}.png")
                cv2.imwrite(frame_path, frame)
                frame_count += 1

                danger_level = random.randint(1, 4)

                await websocket.send_text(json.dumps({"status": "frame received", "danger": danger_level}))
    except WebSocketDisconnect:
        logger.debug("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        await finalize_video(logger)
        logger.debug("Video creation finalized")

def process_nv21_frame(nv21_data, width, height):
    expected_size = width * height * 3 // 2
    if len(nv21_data) != expected_size:
        raise ValueError(f"Invalid frame size: expected {expected_size}, got {len(nv21_data)}")

    yuv_image = np.frombuffer(nv21_data, np.uint8).reshape((height * 3 // 2, width))
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    
    # Show frame to verify it is being processed
    cv2.imshow("Frame", bgr_image)
    cv2.waitKey(1)  # 1ms wait to ensure frame is displayed

    return bgr_image

async def finalize_video(logger):
    global frame_count, width, height

    # Save video to disk
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (width, height))

    for i in range(frame_count):
        frame_path = os.path.join(FRAME_DIR, f"frame_{i:05d}.png")
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            video_writer.write(frame)

    video_writer.release()
    cv2.destroyAllWindows()

    # Clean up temporary frames
    for i in range(frame_count):
        os.remove(os.path.join(FRAME_DIR, f"frame_{i:05d}.png"))

    frame_count = 0
    output_video_path = "output_video.mp4"

    try:
        # Step 1: Upload video to Supabase Storage
        bucket_name = "video"  # Make sure the bucket exists in Supabase Storage
        video_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # Read the video file in binary mode BEFORE the 'with' block ends
        with open(output_video_path, 'rb') as video_file:
            video_data = video_file.read()

        # Upload video to Supabase Storage bucket
        response = supabase.storage.from_("video").upload(
            file=video_data,  # Pass the raw video data
            path=video_name,  # Specify the name to store in Supabase
            file_options={"content-type": "video/mp4"}
        )

        # Check if upload was successful
        if response.status_code != 200:  # Use appropriate status check for Supabase
            logger.error(f"Error uploading video to Supabase: {response.json()}")
            return
        
        # Step 2: Generate video URL
        video_url = f"https://{SUPABASE_URL.replace('https://', '')}/storage/v1/object/public/{bucket_name}/{video_name}"

        # Step 3: Insert video details into the video table
        video_record = {
            "uid": user_id,
            "url": video_url,
            "videoName": video_name,
            "created_at": datetime.utcnow().isoformat()  # Store current time as UTC
        }

        insert_response = supabase.table('video').insert(video_record).execute()

        # Check for errors in insert response
        if insert_response.get('error'):  # Check status for insert operation
            logger.error(f"Error inserting video record into Supabase: {insert_response.json()}")
        else:
            logger.debug("Video record inserted successfully.")

    except Exception as e:
        logger.error(f"Error processing video upload: {str(e)}")