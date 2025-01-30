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
import json
import asyncio

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
    language: Optional[str] = None

class TranslationRequest(BaseModel):
    text: str
    target_language: str

class SentenceGenerationRequest(BaseModel):
    text: str
    target_language: str
    translated_text: str
    initial_language: str
    num_sentences: int = 3

class AnkiExportRequest(BaseModel):
    cards: list[dict]

class CardGenerationRequest(BaseModel):
    text: str
    target_language: str
    voice: str = "alloy"
    generate_original_audio: bool = True
    generate_translation_audio: bool = True
    include_examples: bool = True
    num_sentences: int = 3

class CardGenerationResponse(BaseModel):
    original_text: str
    translated_text: str
    initial_language: str
    target_language: str
    original_audio: Optional[str]
    translation_audio: Optional[str]
    example_sentences: Optional[list[dict]]

class SentenceToCardRequest(BaseModel):
    original_text: str
    translated_text: str
    initial_language: str
    target_language: str
    voice: str = "alloy"
    generate_original_audio: bool = True
    generate_translation_audio: bool = True
    include_examples: bool = True
    num_sentences: int = 3

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
        
        # Set dialect instructions based on language
        dialect_instruction = ""
        if request.language:
            if request.language.lower() == "arabic":
                dialect_instruction = " استخدم لهجة مصرية واضحة. لهجة مصرية أصيلة."  # "Use a clear Egyptian accent. Authentically Egyptian"
        
        # Generate audio from text using the correct API endpoint
        logger.info("Making API request to OpenAI...")
        
        # Modify system prompt for slow speech if requested
        system_prompt = "You are just going to read the text sent by the user out loud. Do not say or do anything besides precisely the text shown here."
        if request.language:
            system_prompt = f"You are just going to read the text sent by the user out loud. The following text is in {request.language}. Do not say or do anything besides precisely the text shown here. Use the correct accent for the given language. {dialect_instruction}"
        if request.slow:
            system_prompt += " Speak extremely slowly and clearly, enunciating each word carefully."
        
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
        
        # Determine if we need to request vowel marks
        needs_vowels = request.target_language.lower() in ["urdu"]
        vowel_instruction = "You must include all short vowel marks/diacritics in the translation." if needs_vowels else ""
        
        # Add Egyptian dialect specification for Arabic
        target_language = request.target_language
        if target_language.lower() == "arabic":
            target_language = "Arabic (Egyptian dialect)"

        completion = client.chat.completions.create(
            model="gpt-4o",
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a translator. First detect the language of the input text, then translate it to {target_language}. {vowel_instruction} Return a JSON object with two fields: 'translated_text' containing only the translation, and 'initial_language' containing the detected language name in English.

Example input: "Bonjour le monde"
Example response: {{"translated_text": "Hello world", "initial_language": "French"}}"""
                },
                {
                    "role": "user",
                    "content": request.text
                }
            ]
        )
        
        response = completion.choices[0].message.content.strip()
        logger.info("Successfully received translation and language detection")
        
        # Parse the JSON response from GPT
        result = json.loads(response)
        
        return JSONResponse(content=result)
    except ValueError as e:
        logger.error(f"ValueError in translation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing GPT response as JSON: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing translation response")
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
                    "content": f"""You are a language learning assistant. Create {request.num_sentences} simple, natural sentences that use or reference the following phrase pair:
Original phrase ({request.initial_language}): "{request.text}"
Translated phrase ({request.target_language}): "{request.translated_text}"

For each sentence, return it in this exact format:
'original: [sentence in {request.initial_language}]\\ntranslated: [sentence in {request.target_language}]'

Make sure each sentence pair naturally incorporates or references the original phrase or its translation."""
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
                orig_audio_field = ''
                trans_audio_field = ''
                
                # Only process original audio if it exists
                if card_data['original_audio']:
                    orig_audio_filename = f'audio_original_{i}.wav'
                    orig_audio_path = os.path.join(temp_dir, orig_audio_filename)
                    with open(orig_audio_path, 'wb') as f:
                        f.write(base64.b64decode(card_data['original_audio']))
                    media_files.append((orig_audio_path, orig_audio_filename))
                    orig_audio_field = f'[sound:{orig_audio_filename}]'
                
                # Only process translation audio if it exists
                if card_data['translation_audio']:
                    trans_audio_filename = f'audio_translated_{i}.wav'
                    trans_audio_path = os.path.join(temp_dir, trans_audio_filename)
                    with open(trans_audio_path, 'wb') as f:
                        f.write(base64.b64decode(card_data['translation_audio']))
                    media_files.append((trans_audio_path, trans_audio_filename))
                    trans_audio_field = f'[sound:{trans_audio_filename}]'
                
                # Create note
                note = genanki.Note(
                    model=model,
                    fields=[
                        card_data['original'],
                        card_data['translation'],
                        card_data['sentences'],
                        orig_audio_field,
                        trans_audio_field
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

@app.post("/api/generate-card")
async def generate_card(
    request: CardGenerationRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    try:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="API key is required")

        if not request.text.strip():
            raise ValueError("Text cannot be empty")

        logger.info(f"Starting card generation for text: {request.text[:50]}...")
        
        # Initialize client with provided API key
        client = get_openai_client(x_api_key)

        # 1. First get translation and language detection
        completion = client.chat.completions.create(
            model="gpt-4o",
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a translator. First detect the language of the input text, then translate it to {request.target_language}. Return a JSON object with two fields: 'translated_text' containing only the translation, and 'initial_language' containing the detected language name in English.

Example input: "Bonjour le monde"
Example response: {{"translated_text": "Hello world", "initial_language": "French"}}"""
                },
                {
                    "role": "user",
                    "content": request.text
                }
            ]
        )
        
        translation_data = json.loads(completion.choices[0].message.content.strip())

        # Initialize response data
        response_data = {
            "original_text": request.text,
            "translated_text": translation_data["translated_text"],
            "initial_language": translation_data["initial_language"],
            "target_language": request.target_language,
            "original_audio": None,
            "translation_audio": None,
            "example_sentences": None
        }

        # 2. Generate audio in parallel if requested
        audio_tasks = []
        
        if request.generate_original_audio:
            original_audio_completion = client.chat.completions.create(
                model="gpt-4o-audio-preview",
                modalities=["text", "audio"],
                audio={"voice": request.voice, "format": "wav"},
                messages=[
                    {
                        "role": "system",
                        "content": f"You are just going to read the text sent by the user out loud. The following text is in {translation_data['initial_language']}. Do not say or do anything besides precisely the text shown here."
                    },
                    {
                        "role": "user",
                        "content": request.text
                    }
                ]
            )
            response_data["original_audio"] = original_audio_completion.choices[0].message.audio.data

        if request.generate_translation_audio:
            translation_audio_completion = client.chat.completions.create(
                model="gpt-4o-audio-preview",
                modalities=["text", "audio"],
                audio={"voice": request.voice, "format": "wav"},
                messages=[
                    {
                        "role": "system",
                        "content": f"You are just going to read the text sent by the user out loud. The following text is in {request.target_language}. Do not say or do anything besides precisely the text shown here."
                    },
                    {
                        "role": "user",
                        "content": translation_data["translated_text"]
                    }
                ]
            )
            response_data["translation_audio"] = translation_audio_completion.choices[0].message.audio.data

        # 3. Generate example sentences if requested
        if request.include_examples:
            sentences_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a language learning assistant. Create {request.num_sentences} simple, natural sentences that use or reference the following phrase pair:
Original phrase ({translation_data['initial_language']}): "{request.text}"
Translated phrase ({request.target_language}): "{translation_data['translated_text']}"

For each sentence, return it in this exact format:
'original: [sentence in {translation_data['initial_language']}]\\ntranslated: [sentence in {request.target_language}]'

Make sure each sentence pair naturally incorporates or references the original phrase or its translation."""
                    },
                    {
                        "role": "user",
                        "content": request.text
                    }
                ]
            )

            # Parse the response into the expected format
            response_text = sentences_completion.choices[0].message.content
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
            
            response_data["example_sentences"] = sentences

        return JSONResponse(content=response_data)

    except ValueError as e:
        logger.error(f"ValueError in card generation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in card generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating card: {str(e)}")

@app.post("/api/sentence-to-card")
async def sentence_to_card(
    request: SentenceToCardRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    try:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="API key is required")

        if not request.original_text.strip() or not request.translated_text.strip():
            raise ValueError("Both original and translated text are required")

        logger.info(f"Starting sentence-to-card conversion for text: {request.original_text[:50]}...")
        
        # Initialize client with provided API key
        client = get_openai_client(x_api_key)

        # Initialize response data
        response_data = {
            "original_text": request.original_text,
            "translated_text": request.translated_text,
            "initial_language": request.initial_language,
            "target_language": request.target_language,
            "original_audio": None,
            "translation_audio": None,
            "example_sentences": None
        }

        # Generate audio if requested
        if request.generate_original_audio:
            original_audio_completion = client.chat.completions.create(
                model="gpt-4o-audio-preview",
                modalities=["text", "audio"],
                audio={"voice": request.voice, "format": "wav"},
                messages=[
                    {
                        "role": "system",
                        "content": f"You are just going to read the text sent by the user out loud. The following text is in {request.initial_language}. Do not say or do anything besides precisely the text shown here."
                    },
                    {
                        "role": "user",
                        "content": request.original_text
                    }
                ]
            )
            response_data["original_audio"] = original_audio_completion.choices[0].message.audio.data

        if request.generate_translation_audio:
            translation_audio_completion = client.chat.completions.create(
                model="gpt-4o-audio-preview",
                modalities=["text", "audio"],
                audio={"voice": request.voice, "format": "wav"},
                messages=[
                    {
                        "role": "system",
                        "content": f"You are just going to read the text sent by the user out loud. The following text is in {request.target_language}. Do not say or do anything besides precisely the text shown here."
                    },
                    {
                        "role": "user",
                        "content": request.translated_text
                    }
                ]
            )
            response_data["translation_audio"] = translation_audio_completion.choices[0].message.audio.data

        # Generate example sentences if requested
        if request.include_examples:
            sentences_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a language learning assistant. Create {request.num_sentences} simple, natural sentences that use or reference the following phrase pair:
Original phrase ({request.initial_language}): "{request.original_text}"
Translated phrase ({request.target_language}): "{request.translated_text}"

For each sentence, return it in this exact format:
'original: [sentence in {request.initial_language}]\\ntranslated: [sentence in {request.target_language}]'

Make sure each sentence pair naturally incorporates or references the original phrase or its translation."""
                    },
                    {
                        "role": "user",
                        "content": request.original_text
                    }
                ]
            )

            # Parse the response into the expected format
            response_text = sentences_completion.choices[0].message.content
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
            
            response_data["example_sentences"] = sentences

        return JSONResponse(content=response_data)

    except ValueError as e:
        logger.error(f"ValueError in sentence-to-card conversion: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in sentence-to-card conversion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error converting sentence to card: {str(e)}") 