import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from fastapi.responses import JSONResponse
from dotenv import dotenv_values
from pathlib import Path

print("Debug: Environment variables:")
print(os.environ)
print("\nDebug: Loading .env from:", Path(__file__).parent / '.env')
# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
config = dotenv_values(env_path)
api_key = config.get('OPENAI_API_KEY')

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize OpenAI client with explicit API key from .env
client = OpenAI(api_key=api_key.strip())

app = FastAPI(title="Language Learning API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextToSpeechRequest(BaseModel):
    text: str
    voice: str = "alloy"

@app.get("/")
async def root():
    return {"message": "Welcome to the Language Learning API"}

@app.post("/api/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    try:
        # Generate audio from text
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": request.voice, "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": request.text
                }
            ]
        )

        # Get audio data from response
        audio_data = completion.choices[0].message.audio.data
        
        return JSONResponse(content={
            "audio_data": audio_data,
            "format": "wav"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 