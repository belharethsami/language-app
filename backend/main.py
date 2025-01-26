import base64
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from fastapi.responses import JSONResponse
import logging
from typing import Optional
from dotenv import load_dotenv
import genanki
import tempfile
import zipfile
import io

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
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

class TextToSpeechRequest(BaseModel):
    text: str
    voice: str = "alloy"
    slow: bool = False

class TranslationRequest(BaseModel):
    text: str
    target_language: str

class SentenceGenerationRequest(BaseModel):
    text: str
    target_language: str
    num_sentences: int = 3

class AnkiExportRequest(BaseModel):
    cards: list[dict]

def get_openai_client(api_key: str):
    return OpenAI(api_key=api_key)

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
        logger.info(f"Slow mode: {request.slow}")
        
        # Initialize client with provided API key
        client = get_openai_client(x_api_key)
        
        # Add Arabic dialect specification for Arabic text
        if any(c in 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيپچژکگی' for c in request.text):
            request.text = request.text + " (باللهجة المصرية)"
        
        # Generate audio from text using the correct API endpoint
        logger.info("Making API request to OpenAI...")
        
        # Modify system prompt for slow speech if requested
        system_prompt = "You are just going to transcribe the text sent by the user. do not say or do anything besides precisely the text shown here. use the correct accent for the given language."
        if request.slow:
            system_prompt += " Speak slowly and clearly, enunciating each word carefully."
        
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": request.voice, "format": "wav"},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
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
        
        # Add Egyptian dialect specification for Arabic
        target_language = request.target_language
        if target_language.lower() == "arabic":
            target_language = "Arabic (Egyptian dialect)"

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a translator. Translate the following text to {target_language}. Provide only the translation, no explanations."
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

@app.post("/api/generate-sentences")
async def generate_sentences(
    request: SentenceGenerationRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    try:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="API key is required")

        if not request.text.strip():
            raise ValueError("Text cannot be empty")

        logger.info(f"Starting sentence generation for text: {request.text[:50]}...")
        
        # Initialize client with provided API key
        client = get_openai_client(x_api_key)

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a language learning assistant. Create {request.num_sentences} simple, natural sentences in both languages. For each sentence, return it in this exact format: 'original: [sentence in original language]\\ntranslated: [sentence in {request.target_language}]'"
                },
                {
                    "role": "user",
                    "content": request.text
                }
            ]
        )

        # Parse the response into the expected format
        response_text = completion.choices[0].message.content
        sentences = []
        
        # Split into sentence pairs and parse
        pairs = response_text.strip().split('\n\n')
        for pair in pairs:
            if not pair.strip():
                continue
            lines = pair.strip().split('\n')
            if len(lines) >= 2:
                original = lines[0].replace('original:', '').strip()
                translated = lines[1].replace('translated:', '').strip()
                sentences.append({
                    "original": original,
                    "translated": translated
                })
        
        logger.info("Successfully generated sentences")
        
        return JSONResponse(content={"sentences": sentences})
    except ValueError as e:
        logger.error(f"ValueError in sentence generation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in sentence generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating sentences: {str(e)}")

@app.post("/api/export-anki")
async def export_anki(request: AnkiExportRequest):
    try:
        # Create a unique model ID and deck ID
        model_id = 1607392319
        deck_id = 2059400110

        # Define the note model
        model = genanki.Model(
            model_id,
            'Language Learning Card',
            fields=[
                {'name': 'Original'},
                {'name': 'Translation'},
                {'name': 'Example Sentences'},
                {'name': 'Audio (Original)'},
                {'name': 'Audio (Translation)'}
            ],
            templates=[{
                'name': 'Card 1',
                'qfmt': '{{Original}}<br>{{Audio (Original)}}',
                'afmt': '{{FrontSide}}<hr>{{Translation}}<br>{{Audio (Translation)}}<br><br><div style="font-size: 0.85em;">{{Example Sentences}}</div>',
            }]
        )

        # Create a new deck
        deck = genanki.Deck(deck_id, 'Language Learning Cards')
        
        # Create a temporary directory for media files
        with tempfile.TemporaryDirectory() as temp_dir:
            media_files = []
            
            # Process each card
            for i, card_data in enumerate(request.cards, 1):
                # Save audio files
                orig_audio_filename = f'audio_original_{i}.wav'
                trans_audio_filename = f'audio_translated_{i}.wav'
                
                # Save original audio
                orig_audio_path = os.path.join(temp_dir, orig_audio_filename)
                with open(orig_audio_path, 'wb') as f:
                    f.write(base64.b64decode(card_data['original_audio']))
                media_files.append((orig_audio_path, orig_audio_filename))
                
                # Save translation audio
                trans_audio_path = os.path.join(temp_dir, trans_audio_filename)
                with open(trans_audio_path, 'wb') as f:
                    f.write(base64.b64decode(card_data['translation_audio']))
                media_files.append((trans_audio_path, trans_audio_filename))
                
                # Create note
                note = genanki.Note(
                    model=model,
                    fields=[
                        card_data['original'],
                        card_data['translation'],
                        card_data['sentences'],
                        f'[sound:{orig_audio_filename}]',
                        f'[sound:{trans_audio_filename}]'
                    ]
                )
                deck.add_note(note)
            
            # Create the package
            package = genanki.Package(deck)
            package.media_files = [f[0] for f in media_files]
            
            # Save to a bytes buffer
            buffer = io.BytesIO()
            package.write_to_file(buffer)
            buffer.seek(0)
            
            return Response(
                content=buffer.read(),
                media_type='application/zip',
                headers={
                    'Content-Disposition': 'attachment; filename=language_cards.apkg'
                }
            )
            
    except Exception as e:
        logger.error(f"Error creating Anki package: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating Anki package: {str(e)}") 