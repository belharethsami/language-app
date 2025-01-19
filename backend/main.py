import base64
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from fastapi.responses import JSONResponse
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Language Learning API")

# Configure CORS - Development mode (more permissive)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=False,  # Must be False for allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextToSpeechRequest(BaseModel):
    text: str
    voice: str = "alloy"

class TranslationRequest(BaseModel):
    text: str
    target_language: str

def get_openai_client(api_key: str):
    try:
        # Basic configuration
        config = {
            "api_key": api_key,
            "base_url": "https://api.openai.com/v1",
            "timeout": 60,  # seconds
            "max_retries": 2
        }
        
        return OpenAI(**config)
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing OpenAI client: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Language Learning API"}

@app.post("/api/text-to-speech")
async def text_to_speech(
    request: TextToSpeechRequest, 
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    try:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="API key is required")

        if not request.text.strip():
            raise ValueError("Text cannot be empty")

        logger.info(f"Starting text-to-speech request for text: {request.text[:50]}...")
        logger.info(f"Using voice: {request.voice}")
        
        # Initialize client with provided API key
        client = get_openai_client(x_api_key)
        
        # Generate audio from text using the correct API endpoint
        logger.info("Making API request to OpenAI...")
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": request.voice, "format": "wav"},
            messages=[
                {
                    "role": "system",
                    "content": "You are just going to transcribe the text sent by the user. do not say or do anything besides precisely the text shown here. use the correct accent for the given language."
                },
                {
                    "role": "user",
                    "content": request.text
                }
            ]
        )
        logger.info("Received response from OpenAI")

        # Get audio data from response
        audio_data = completion.choices[0].message.audio.data
        logger.info("Successfully extracted audio data")
        
        return JSONResponse(content={
            "audio_data": audio_data,
            "format": "wav"
        })
    except ValueError as e:
        logger.error(f"ValueError in text-to-speech: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in text-to-speech: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.post("/api/translate")
async def translate(
    request: TranslationRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    try:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="API key is required")

        if not request.text.strip():
            raise ValueError("Text cannot be empty")

        logger.info(f"Starting translation request for text: {request.text[:50]}...")
        logger.info(f"Target language: {request.target_language}")

        # Initialize client with provided API key
        client = get_openai_client(x_api_key)

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a translator. Translate the following text to {request.target_language}. Provide only the translation, no explanations."
                },
                {
                    "role": "user",
                    "content": request.text
                }
            ]
        )

        translation = completion.choices[0].message.content.strip()
        logger.info("Successfully received translation")
        
        return JSONResponse(content={
            "translation": translation
        })
    except ValueError as e:
        logger.error(f"ValueError in translation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in translation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error translating text: {str(e)}") 