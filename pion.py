from fastapi import FastAPI, Request
import requests

app = FastAPI()

# 將 SDP offer 發送給 Pion 並返回 SDP answer
@app.post("/sdp")
async def exchange_sdp(request: Request):
    body = await request.body()
    print("Received SDP offer:", body.decode())
    
    pion_response = requests.post("http://localhost:8080/sdp", data=body)
    print("Pion response status:", pion_response.status_code)
    print("Pion response body:", pion_response.text)
    
    if pion_response.status_code == 200:
        return {"sdp": pion_response.text}
    else:
        return {"error": "Failed to communicate with Pion WebRTC server"}


# 啟動 FastAPI 服務器： uvicorn filename:app --reload
