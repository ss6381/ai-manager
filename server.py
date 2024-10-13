import asyncio
import logging
import os

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import outspeed as sp
from avatar.run import VoiceBot

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize VoiceBot
voice_bot = VoiceBot()

@app.on_event("startup")
async def startup_event():
    await voice_bot.setup()

@app.on_event("shutdown")
async def shutdown_event():
    await voice_bot.teardown()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    audio_input_queue = sp.AudioStream()
    text_input_queue = sp.TextStream()
    
    audio_output_stream, text_output_stream = await voice_bot.run(audio_input_queue, text_input_queue)
    
    async def receive():
        while True:
            try:
                data = await websocket.receive_bytes()
                await audio_input_queue.put(data)
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
                break

    async def send():
        async for audio_chunk in audio_output_stream:
            try:
                await websocket.send_bytes(audio_chunk)
            except Exception as e:
                logging.error(f"Error sending audio: {e}")
                break

    receive_task = asyncio.create_task(receive())
    send_task = asyncio.create_task(send())

    await asyncio.gather(receive_task, send_task)

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

