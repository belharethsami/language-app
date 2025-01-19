import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from fastapi.responses import JSONResponse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Language Learning API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client with API key directly
client = OpenAI(api_key="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")

class TextToSpeechRequest(BaseModel):
    text: str
    voice: str = "alloy"

class TranslationRequest(BaseModel):
    text: str
    target_language: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Language Learning API"}

@app.post("/api/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    try:
        if not request.text.strip():
            raise ValueError("Text cannot be empty")

        logger.info(f"Starting text-to-speech request for text: {request.text[:50]}...")
        logger.info(f"Using voice: {request.voice}")
        
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
async def translate(request: TranslationRequest):
    try:
        if not request.text.strip():
            raise ValueError("Text cannot be empty")

        logger.info(f"Starting translation request for text: {request.text[:50]}...")
        logger.info(f"Target language: {request.target_language}")

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