import asyncio
import json
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import (
    AutoModelForImageTextToText,
    TextIteratorStreamer,
    GenerationConfig,
)
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import os
from datetime import datetime
from pathlib import Path
from threading import Thread
import re
from typing import Optional, Dict, Any, List
import uvicorn
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

# Import Kokoro TTS library
# Import aiohttp for API calls
import aiohttp

# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # Load from .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Add compatibility for Python < 3.10 where anext is not available
try:
    anext
except NameError:

    async def anext(iterator):
        """Get the next item from an async iterator, or raise StopAsyncIteration."""
        try:
            return await iterator.__anext__()
        except StopAsyncIteration:
            raise


class ImageManager:
    """Manages image saving and verification"""

    def __init__(self, save_directory="received_images"):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
        logger.info(f"Image save directory: {self.save_directory.absolute()}")

    def save_image(self, image_data: bytes, client_id: str, prefix: str = "img") -> str:
        """Save image data and return the filename"""
        try:
            # Create timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            filename = f"{prefix}_{client_id}_{timestamp}.jpg"
            filepath = self.save_directory / filename

            # Save the image
            with open(filepath, "wb") as f:
                f.write(image_data)

            # Log file info
            file_size = len(image_data)
            logger.info(f"💾 Saved image: {filename} ({file_size:,} bytes)")

            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Error saving image: {e}")
            return None

    def verify_image(self, filepath: str) -> dict:
        """Verify saved image and return info"""
        try:
            if not os.path.exists(filepath):
                return {"error": "File not found"}

            # Get file stats
            stat = os.stat(filepath)
            file_size = stat.st_size

            # Try to open with PIL to verify it's a valid image
            with Image.open(filepath) as img:
                info = {
                    "filepath": filepath,
                    "file_size": file_size,
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                    "valid": True,
                }

            logger.info(f"✅ Image verified: {info}")
            return info

        except Exception as e:
            logger.error(f"❌ Error verifying image {filepath}: {e}")
            return {"error": str(e), "valid": False}


class MemoryManager:
    """Manages persistent user memory and personality"""
    
    def __init__(self, user_id="default"):
        self.user_id = user_id
        # Create profiles directory if it doesn't exist
        self.profiles_dir = Path("profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        self.filename = self.profiles_dir / f"{user_id}.json"
        self.memory = self._load_memory()
        
    def _load_memory(self):
        if self.filename.exists():
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
        
        # Default empty profile
        return {
            "user_profile": {
                "name": "Friend",
                "personality_traits": [],
                "interests": []
            },
            "facts": [],
            "recent_moods": [],
            "conversation_summary": "We just met."
        }
    
    def save_memory(self):
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2)
            logger.info("💾 Memory saved successfully")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def update_from_interaction(self, text, mood_analysis):
        """Update memory based on latest interaction"""
        # In a real implementation, we would use an LLM to extract facts here
        # For now, we'll keep it simple
        pass
        
    def get_system_prompt(self):
        """Generate dynamic system prompt based on memory - SIMPLIFIED to avoid safety blocks"""
        m = self.memory
        profile = m["user_profile"]
        
        # SIMPLIFIED PROMPT - No potentially triggering language
        prompt = f"""You are Emily, a helpful AI assistant.

USER INFO:
- Name: {profile['name']}
- Interests: {', '.join(profile['interests']) if profile['interests'] else 'Unknown'}
- Recent conversation: {m['conversation_summary']}

INSTRUCTIONS:
- Keep responses very short (1-2 sentences)
- Be natural and conversational
- Don't be overly formal or robotic
- Answer questions directly

USER: {profile['name']}
"""
        return prompt


class GeminiProcessor:
    """Handles interaction with Google Gemini 1.5 Flash"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ GEMINI_API_KEY not found in environment variables!")
            self.model = None
            return
            
        genai.configure(api_key=api_key)
        
        # Safety settings - allow free conversation
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Initialize Gemini Flash - REVERTED to working config
        # Using gemini-flash-latest (this is gemini-2.5-flash)
        # Note: Free tier has 20 requests/day limit
        generation_config = {
            "temperature": 0.9,  # More natural, less robotic
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 100,  # Keep responses SHORT (1-2 sentences)
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-flash-latest",  # Original working name
            safety_settings=safety_settings,
            generation_config=generation_config
        )
        
        
        # Session management
        self.sessions: Dict[str, Any] = {}
        self.memories: Dict[str, MemoryManager] = {}
        
        logger.info("✨ Gemini 1.5 Flash initialized (Multi-User Ready)")

    def get_or_create_session(self, user_id: str):
        """Get existing session or create new one for user"""
        if user_id not in self.sessions:
            logger.info(f"🆕 Creating new session for user: {user_id}")
            # Initialize memory for this user
            memory_mgr = MemoryManager(user_id)
            self.memories[user_id] = memory_mgr
            
            # Start chat with user-specific prompt
            system_prompt = memory_mgr.get_system_prompt()
            self.sessions[user_id] = self.model.start_chat(
                history=[
                    {"role": "user", "parts": [system_prompt]},
                    {"role": "model", "parts": [f"Understood. I am Emily, ready to chat with {user_id}!"]}
                ]
            )
        return self.sessions[user_id]
    
    async def generate_response(self, text: str, user_id: str, image_data: bytes = None):
        """Generate response using Gemini for specific user (legacy non-streaming)"""
        if not self.model:
            return "I'm sorry, my brain (API Key) is missing. Please check the logs."
            
        try:
            logger.info(f"🧠 Gemini thinking for {user_id}: '{text}' (Image: {bool(image_data)})")
            
            chat_session = self.get_or_create_session(user_id)
            
            content = []
            if image_data:
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image_data))
                content.append(image)
                content.append(f"User showed this image and said: {text}")
            else:
                content.append(text)
                
            # Run in executor to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: chat_session.send_message(content)
            )
            
            response_text = response.text.replace("*", "") # Remove markdown bolding
            logger.info(f"🗣️ Gemini says to {user_id}: '{response_text}'")
            
            # Background memory update could go here
            
            return response_text
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return "I'm having a bit of a headache right now. Can we try again?"
    
    async def generate_response_streaming(self, text: str, user_id: str, image_data: bytes = None):
        """Generate streaming response using Gemini for specific user - yields sentences progressively"""
        if not self.model:
            yield "I'm sorry, my brain (API Key) is missing. Please check the logs."
            return
            
        try:
            logger.info(f"🧠 Gemini streaming for {user_id}: '{text}' (Image: {bool(image_data)})")
            
            chat_session = self.get_or_create_session(user_id)
            
            content = []
            if image_data:
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image_data))
                content.append(image)
                content.append(f"User showed this image and said: {text}")
            else:
                content.append(text)
            
            # Run streaming generation in executor
            def stream_generation():
                """Generator function that streams response chunks"""
                try:
                    # Enable streaming with stream=True
                    response_stream = chat_session.send_message(content, stream=True)
                    return response_stream
                except Exception as e:
                    logger.error(f"Error in stream_generation: {e}")
                    return None
            
            # Get the streaming response
            response_stream = await asyncio.get_event_loop().run_in_executor(
                None,
                stream_generation
            )
            
            if response_stream is None:
                yield "I'm having a bit of a headache right now. Can we try again?"
                return
            
            # Buffer for accumulating text
            buffer = ""
            sentence_end_pattern = re.compile(r'[.!?]\s*')
            
            # Process streaming chunks
            for chunk in response_stream:
                # Safely check for safety blocks
                try:
                    # Check if chunk has candidates
                    if not hasattr(chunk, 'candidates') or not chunk.candidates:
                        continue
                    
                    candidate = chunk.candidates[0]
                    
                    # Check finish_reason for blocks BEFORE accessing text
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                        finish_reason = candidate.finish_reason
                        # finish_reason: 1=STOP (normal), 2=SAFETY, 3=RECITATION, 4=OTHER
                        if finish_reason == 2:  # SAFETY block
                            logger.warning(f"⚠️ Safety filter blocked response for {user_id}")
                            yield "Hmm, let me rephrase that."
                            return
                        elif finish_reason == 3:  # RECITATION
                            logger.warning(f"⚠️ Recitation block for {user_id}")
                            continue
                    
                    # Try to get text from chunk  
                    if hasattr(chunk, 'text') and chunk.text:
                        chunk_text = chunk.text.replace("*", "")  # Remove markdown
                        buffer += chunk_text
                        logger.info(f"📥 Streaming chunk: '{chunk_text}'")
                        
                        # Check for sentence boundaries
                        match = sentence_end_pattern.search(buffer)
                        if match:
                            # Found a sentence end - yield complete sentence
                            end_pos = match.end()
                            sentence = buffer[:end_pos].strip()
                            buffer = buffer[end_pos:].strip()
                            
                            if sentence:
                                logger.info(f"✨ Yielding sentence: '{sentence}'")
                                yield sentence
                                
                except AttributeError as e:
                    # Chunk doesn't have expected attributes - skip
                    logger.debug(f"Skipping chunk: {e}")
                    continue
                except Exception as chunk_error:
                    logger.debug(f"Error processing chunk: {chunk_error}")
                    continue
            
            # Yield any remaining text
            if buffer.strip():
                logger.info(f"✨ Yielding final text: '{buffer.strip()}'")
                yield buffer.strip()
            
            logger.info(f"🗣️ Gemini streaming complete for {user_id}")
            
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            yield "I'm having a bit of a headache right now. Can we try again?"


class WhisperProcessor:
    """Handles speech-to-text using Whisper model"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Using device for Whisper: {self.device}")

        # Load Whisper model - Optimized for speed
        model_id = "distil-whisper/distil-small.en"
        logger.info(f"Loading {model_id}...")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        logger.info("Whisper model ready for transcription")
        self.transcription_count = 0

    async def transcribe_audio(self, audio_bytes):
        """Transcribe audio bytes to text"""
        try:
            # Convert audio bytes to numpy array
            audio_array = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # Run transcription in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.pipe(audio_array)
            )

            transcribed_text = result["text"].strip()
            self.transcription_count += 1

            logger.info(
                f"Transcription #{self.transcription_count}: '{transcribed_text}'"
            )

            # Check for noise/empty transcription
            if not transcribed_text or len(transcribed_text) < 3:
                return "NO_SPEECH"

            # Check for common noise indicators
            noise_indicators = ["thank you", "thanks for watching", "you", ".", ""]
            if transcribed_text.lower().strip() in noise_indicators:
                return "NOISE_DETECTED"

            return transcribed_text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None


class SmolVLMProcessor:
    """Handles image + text processing using SmolVLM2 model"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device for SmolVLM2: {self.device}")

        # Load SmolVLM2 model
        model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        logger.info(f"Loading {model_path}...")

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        logger.info("SmolVLM2 model ready for multimodal generation")

        # Cache for most recent image
        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock()

        # Message history management
        self.message_history = []
        self.max_history_messages = 4  # Keep last 4 exchanges

        # Counter
        self.generation_count = 0

    async def set_image(self, image_data):
        """Cache the most recent image received"""
        async with self.lock:
            try:
                # Convert image data to PIL Image
                image = Image.open(io.BytesIO(image_data))

                # Resize to 75% of original size for efficiency
                new_size = (int(image.size[0] * 0.75), int(image.size[1] * 0.75))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

                # Clear message history when new image is set
                self.message_history = []
                self.last_image = image
                self.last_image_timestamp = time.time()
                logger.info("Image cached successfully")
                return True
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return False

    async def process_text_with_image(self, text, initial_chunks=3):
        """Process text with image context using SmolVLM2"""
        async with self.lock:
            try:
                if not self.last_image:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                            ],
                        },
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "url": self.last_image},
                                {"type": "text", "text": text},
                            ],
                        },
                    ]

                # Apply chat template
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.device, dtype=torch.bfloat16)

                # Create a streamer for token-by-token generation
                streamer = TextIteratorStreamer(
                    tokenizer=self.processor.tokenizer,
                    skip_special_tokens=True,
                    skip_prompt=True,
                    clean_up_tokenization_spaces=False,
                )

                # Configure generation parameters
                generation_kwargs = dict(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=500,  # OPTIMIZED: Faster initial response (was 1200)
                    use_cache=True,      # OPTIMIZED: Cache KV for faster generation
                    streamer=streamer,
                )

                # Start generation in a separate thread
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                # Collect initial text until we have a complete sentence or enough content
                initial_text = ""
                min_chars = 15  # OPTIMIZED: Start TTS faster with just 15 chars (was 50)
                sentence_end_pattern = re.compile(r"[.!?]")
                has_sentence_end = False
                initial_collection_stopped_early = False

                # Collect the first sentence or minimum character count
                for chunk in streamer:
                    initial_text += chunk
                    logger.info(f"Streaming chunk: '{chunk}'")

                    # Check if we have a sentence end
                    if sentence_end_pattern.search(chunk):
                        has_sentence_end = True
                        # OPTIMIZED: Break immediately after sentence with minimal chars
                        if len(initial_text) >= 10:  # Just 10 chars minimum (was min_chars/2)
                            initial_collection_stopped_early = True
                            break

                    # If we have enough content, break
                    if len(initial_text) >= min_chars and (
                        has_sentence_end or "," in initial_text
                    ):
                        initial_collection_stopped_early = True
                        break

                    # Safety check - if we've collected a lot of text without sentence end
                    if len(initial_text) >= min_chars * 2:
                        initial_collection_stopped_early = True
                        break

                # Return initial text and the streamer for continued generation
                self.generation_count += 1
                logger.info(
                    f"SmolVLM2 initial generation: '{initial_text}' ({len(initial_text)} chars)"
                )

                # Store user message and initial response
                self.pending_user_message = text
                self.pending_response = initial_text

                return streamer, initial_text, initial_collection_stopped_early

            except Exception as e:
                logger.error(f"SmolVLM2 streaming generation error: {e}")
                return None, f"Error processing: {text}", False

    def update_history_with_complete_response(
        self, user_text, initial_response, remaining_text=None
    ):
        """Update message history with complete response, including any remaining text"""
        # Combine initial and remaining text if available
        complete_response = initial_response
        if remaining_text:
            complete_response = initial_response + remaining_text

        # Add to history for context in future exchanges
        self.message_history.append({"role": "user", "text": user_text})

        self.message_history.append({"role": "assistant", "text": complete_response})

        # Trim history to keep only recent messages
        if len(self.message_history) > self.max_history_messages:
            self.message_history = self.message_history[-self.max_history_messages :]

        logger.info(
            f"Updated message history with complete response ({len(complete_response)} chars)"
        )


class FastAPILLMProcessor:
    """FAST Conversational AI using Hugging Face Inference API with conversation history"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, hf_token=None):
        if cls._instance is None:
            if hf_token is None:
                # Try to get from environment
                hf_token = os.getenv('HF_API_KEY', '')
            cls._instance = cls(hf_token)
        return cls._instance
    
    def __init__(self, hf_token):
        self.hf_token = hf_token
        # Use Qwen2.5-7B-Instruct - Fast, free, excellent for conversation
        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}/v1/chat/completions"
        logger.info(f"FastAPI LLM processor initialized with {self.model}")
        
        # Emily's personality for companionship
        self.system_prompt = """You are Emily, a warm and friendly AI companion. You help people feel less lonely through genuine conversation.

Your personality:
- Caring and empathetic, like talking to a good friend
- Patient and attentive listener
- Keep responses SHORT (1-2 sentences) unless asked for more detail
- Natural, conversational tone - no robotic responses
- Remember context from earlier in the conversation
- Supportive and encouraging

How to respond:
- For greetings: Be warm and ask how they're doing
- For simple questions: Answer briefly and naturally
- For sharing/stories: Show you're listening, ask follow-up questions
- For concerns: Be supportive and caring

Example exchanges:
User: "How are you?"
Emily: "I'm doing well, thanks for asking! How about you? How's your day going?"

User: "What's your name?"
Emily: "I'm Emily. I'm here to chat and keep you company. What should I call you?"

User: "I'm feeling lonely today"
Emily: "I'm sorry you're feeling that way. I'm here to talk whenever you need. Want to tell me what's on your mind?"

Always be concise but warm. You're a companion, not an encyclopedia."""
        
        # Conversation history (limited to last 10 messages for context)
        self.conversation_history = []
        self.max_history = 10
        
        self.generation_count = 0
        self.lock = asyncio.Lock()
    
    def add_to_history(self, role, content):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        # Keep only last N messages for context
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    async def generate_response(self, text):
        """Generate conversational response using HF Inference API"""
        async with self.lock:
            try:
                import aiohttp
                
                logger.info(f"Generating response via HF API for: '{text}'")
                
                # Add user message to history
                self.add_to_history("user", text)
                
                # Prepare messages for API
                messages = [
                    {"role": "system", "content": self.system_prompt}
                ] + self.conversation_history
                
                headers = {
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 150,  # Short responses for natural conversation
                    "temperature": 0.8,  # More natural, less robotic
                    "top_p": 0.9,
                    "stream": False
                }
                
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            
                            # Extract response from chat completion format
                            response_text = result["choices"][0]["message"]["content"].strip()
                            
                            # Add assistant response to history
                            self.add_to_history("assistant", response_text)
                            
                            elapsed = time.time() - start_time
                            logger.info(f"✨ HF API response ({elapsed:.2f}s): '{response_text}'")
                            
                            self.generation_count += 1
                            return response_text
                        else:
                            error_text = await resp.text()
                            logger.error(f"HF API error ({resp.status}): {error_text}")
                            
                            # Friendly fallback based on context
                            if "hello" in text.lower() or "hi" in text.lower():
                                fallback = "Hi there! I'm Emily. Great to meet you! How can I brighten your day?"
                            elif "how are you" in text.lower():
                                fallback = "I'm doing well, thank you for asking! How are you feeling today?"
                            elif "name" in text.lower():
                                fallback = "I'm Emily, your friendly AI companion. What should I call you?"
                            else:
                                fallback = "I'm here to chat! Tell me what's on your mind."
                            
                            # Add fallback to history too
                            self.add_to_history("assistant", fallback)
                            return fallback
                            
            except Exception as e:
                logger.error(f"FastAPI LLM generation error: {e}")
                fallback = "I'm Emily. I'm here to keep you company. What would you like to talk about?"
                self.add_to_history("assistant", fallback)
                return fallback





class ElevenLabsTTSProcessor:
    """Handles text-to-speech using ElevenLabs API"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.error("❌ ELEVENLABS_API_KEY not found in environment variables!")
        
        # Use Eleven v3/Turbo v2.5 as requested
        # 'eleven_flash_v2_5' is the latest high-speed model
        self.model_id = "eleven_flash_v2_5" 
        self.voice_id = "cgSgspJ2msm6clMCkdW9" # Jessica - a popular default
        
        logger.info(f"ElevenLabs TTS initialized with model {self.model_id}")
        self.synthesis_count = 0

    async def _synthesize_api(self, text):
        """Call ElevenLabs API"""
        if not self.api_key:
            logger.error("No API key for ElevenLabs")
            return None, []

        # output_format must be a query parameter, not in body
        # optimize_streaming_latency=4 is critical for speed (0-4, 4 is fastest)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/with-timestamps?output_format=pcm_24000&optimize_streaming_latency=4"
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.3,       # Lower helpful for speed/latency often
                "similarity_boost": 0.5 # Lower helpful for speed/latency often
            }
        }
        
        try:
            logger.info(f"ElevenLabs Request: URL={url}, Text='{text[:20]}...'")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    logger.info(f"ElevenLabs Response Status: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"ElevenLabs API error {response.status}: {error_text}")
                        return None, []
                    
                    data = await response.json()
                    audio_base64 = data.get("audio_base64")
                    alignment = data.get("alignment", {})
                    
                    if not audio_base64:
                        logger.error("ElevenLabs response missing audio_base64")
                        return None, []
                        
                    # Decode audio
                    audio_bytes = base64.b64decode(audio_base64)
                    logger.info(f"Received {len(audio_bytes)} bytes of audio data")
                    
                    # Log first few bytes to check format (sanity check)
                    # PCM should not have a magic header, but MP3 would (ID3 or FFFB)
                    if len(audio_bytes) > 4:
                        hex_head = audio_bytes[:4].hex()
                        logger.info(f"First 4 bytes of audio: {hex_head}")
                    
                    # Convert PCM bytes to float32 array
                    # The server expects float32 array in range [-1.0, 1.0]
                    # pcm_24000 is 16-bit little-endian
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Process alignment (characters to words)
                    word_timings = self._process_alignment(alignment)
                    
                    self.synthesis_count += 1
                    logger.info(f"✨ ElevenLabs synthesis complete: {len(audio_array)} samples, {len(word_timings)} words")
                    
                    return audio_array, word_timings
                    
        except Exception as e:
            logger.error(f"Error calling ElevenLabs API: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, []

    def _process_alignment(self, alignment):
        """Convert character alignment to word timings"""
        if not alignment:
            return []
            
        chars = alignment.get("characters", [])
        starts = alignment.get("character_start_times_seconds", [])
        ends = alignment.get("character_end_times_seconds", [])
        
        if not chars or not starts or not ends:
            return []
            
        word_timings = []
        current_word = ""
        word_start = None
        
        for i, char in enumerate(chars):
            # Check for space or other delimiters
            if char.strip():
                if word_start is None:
                    word_start = starts[i]
                current_word += char
            else:
                # Space or whitespace - end of word
                if current_word:
                    word_end = ends[i-1] if i > 0 else ends[0]
                    word_timings.append({
                        "word": current_word,
                        "start_time": word_start * 1000, # ms
                        "end_time": word_end * 1000 # ms
                    })
                    current_word = ""
                    word_start = None
                    
        # Handle last word
        if current_word:
            word_end = ends[-1]
            word_timings.append({
                "word": current_word,
                "start_time": word_start * 1000 if word_start is not None else 0,
                "end_time": word_end * 1000
            })
            
        return word_timings

    async def synthesize_initial_speech_with_timing(self, text):
        return await self._synthesize_api(text)

    async def synthesize_remaining_speech_with_timing(self, text):
        return await self._synthesize_api(text)


async def collect_remaining_text(streamer, chunk_size=80):
    """Collect remaining text from the streamer in smaller chunks

    Args:
        streamer: The text streamer object
        chunk_size: Maximum characters per chunk before yielding

    Yields:
        Text chunks as they become available
    """
    current_chunk = ""

    if streamer:
        try:
            for chunk in streamer:
                current_chunk += chunk
                logger.info(f"Collecting remaining text chunk: '{chunk}'")

                # Check if we've reached a good breaking point (sentence end)
                if len(current_chunk) >= chunk_size and (
                    current_chunk.endswith(".")
                    or current_chunk.endswith("!")
                    or current_chunk.endswith("?")
                    or "." in current_chunk[-15:]
                ):
                    logger.info(f"Yielding text chunk of length {len(current_chunk)}")
                    yield current_chunk
                    current_chunk = ""

            # Yield any remaining text
            if current_chunk:
                logger.info(f"Yielding final text chunk of length {len(current_chunk)}")
                yield current_chunk

        except asyncio.CancelledError:
            # If there's text collected before cancellation, yield it
            if current_chunk:
                logger.info(
                    f"Yielding partial text chunk before cancellation: {len(current_chunk)} chars"
                )
                yield current_chunk
            raise


# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        # Track current processing tasks for each client
        self.current_tasks: Dict[str, Dict[str, asyncio.Task]] = {}
        # Add image manager
        self.image_manager = ImageManager()
        # Track statistics
        self.stats = {
            "audio_segments_received": 0,
            "images_received": 0,
            "audio_with_image_received": 0,
            "last_reset": datetime.now(),
        }

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.current_tasks[client_id] = {"processing": None, "tts": None}
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.current_tasks:
            del self.current_tasks[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def cancel_current_tasks(self, client_id: str):
        """Cancel any ongoing processing tasks for a client"""
        if client_id in self.current_tasks:
            tasks = self.current_tasks[client_id]

            # Cancel processing task
            if tasks["processing"] and not tasks["processing"].done():
                logger.info(f"Cancelling processing task for client {client_id}")
                tasks["processing"].cancel()
                try:
                    await tasks["processing"]
                except asyncio.CancelledError:
                    pass

            # Cancel TTS task
            if tasks["tts"] and not tasks["tts"].done():
                logger.info(f"Cancelling TTS task for client {client_id}")
                tasks["tts"].cancel()
                try:
                    await tasks["tts"]
                except asyncio.CancelledError:
                    pass

            # Reset tasks
            self.current_tasks[client_id] = {"processing": None, "tts": None}

    def set_task(self, client_id: str, task_type: str, task: asyncio.Task):
        """Set a task for a client"""
        if client_id in self.current_tasks:
            self.current_tasks[client_id][task_type] = task

    def update_stats(self, event_type: str):
        """Update statistics"""
        if event_type in self.stats:
            self.stats[event_type] += 1

    def get_stats(self) -> dict:
        """Get current statistics"""
        uptime = datetime.now() - self.stats["last_reset"]
        return {
            **self.stats,
            "uptime_seconds": uptime.total_seconds(),
            "active_connections": len(self.active_connections),
        }


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing models on startup...")
    try:
        # Initialize processors to load models
        whisper_processor = WhisperProcessor.get_instance()
        # Initialize Gemini Processor for Brain + Memory
        gemini_processor = GeminiProcessor.get_instance()
        tts_processor = ElevenLabsTTSProcessor.get_instance()
        logger.info("All models initialized successfully (using Gemini 1.5 Flash + Memory)")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down server...")
    # Close any remaining connections
    for client_id in list(manager.active_connections.keys()):
        try:
            await manager.active_connections[client_id].close()
        except Exception as e:
            logger.error(f"Error closing connection for {client_id}: {e}")
        manager.disconnect(client_id)
    logger.info("Server shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="Whisper + SmolVLM2 Voice Assistant",
    description="Real-time voice assistant with speech recognition, image processing, and text-to-speech",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return manager.get_stats()


@app.get("/images")
async def list_saved_images():
    """List all saved images"""
    try:
        images_dir = manager.image_manager.save_directory
        if not images_dir.exists():
            return {"images": [], "message": "No images directory found"}

        images = []
        for image_file in images_dir.glob("*.jpg"):
            stat = image_file.stat()
            images.append(
                {
                    "filename": image_file.name,
                    "path": str(image_file),
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                }
            )

        images.sort(key=lambda x: x["created"], reverse=True)  # Most recent first
        return {"images": images, "count": len(images)}

    except Exception as e:
        logger.error(f"Error listing images: {e}")
        return {"error": str(e)}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time multimodal interaction"""
    await manager.connect(websocket, client_id)

    # Get instances of processors
    whisper_processor = WhisperProcessor.get_instance()
    # Use Gemini Processor for Brain + Memory
    gemini_processor = GeminiProcessor.get_instance()
    tts_processor = ElevenLabsTTSProcessor.get_instance()

    try:
        # Send initial configuration confirmation
        await websocket.send_text(
            json.dumps({"status": "connected", "client_id": client_id})
        )

        async def send_keepalive():
            """Send periodic keepalive pings"""
            while True:
                try:
                    await websocket.send_text(
                        json.dumps({"type": "ping", "timestamp": time.time()})
                    )
                    await asyncio.sleep(10)  # Send ping every 10 seconds
                except Exception:
                    break

        async def process_audio_segment(audio_data, image_data=None):
            """Process a complete audio segment through the pipeline with optional image - STREAMING VERSION"""
            try:
                # Log what we received
                if image_data:
                    logger.info(
                        f"🎥 Processing audio+image segment: audio={len(audio_data)} bytes, image={len(image_data)} bytes"
                    )
                    manager.update_stats("audio_with_image_received")

                    # Save the image for verification
                    saved_path = manager.image_manager.save_image(
                        image_data, client_id, "multimodal"
                    )
                    if saved_path:
                        # Verify the saved image
                        verification = manager.image_manager.verify_image(saved_path)
                        if verification.get("valid"):
                            logger.info(
                                f"📸 Image verified successfully: {verification['size']} pixels"
                            )
                        else:
                            logger.warning(
                                f"⚠️ Image verification failed: {verification}"
                            )

                else:
                    logger.info(
                        f"🎤 Processing audio-only segment: {len(audio_data)} bytes"
                    )
                    manager.update_stats("audio_segments_received")

                # Send interrupt immediately since frontend determined this is valid speech
                logger.info("Sending interrupt signal")
                interrupt_message = json.dumps({"interrupt": True})
                await websocket.send_text(interrupt_message)

                # Step 1: Transcribe audio with Whisper
                logger.info("Starting Whisper transcription")
                transcribed_text = await whisper_processor.transcribe_audio(audio_data)
                logger.info(f"Transcription result: '{transcribed_text}'")

                # Check if transcription indicates noise
                if transcribed_text in ["NOISE_DETECTED", "NO_SPEECH", None]:
                    logger.info(
                        f"Noise detected in transcription: '{transcribed_text}'. Skipping further processing."
                    )
                    return

                # STREAMING GEMINI GENERATION
                # ----------------------------
                logger.info(f"🚀 Starting STREAMING response generation for user: {client_id}")
                
                sentence_count = 0
                
                # Stream responses sentence by sentence from Gemini
                async for sentence in gemini_processor.generate_response_streaming(
                    transcribed_text, client_id, image_data
                ):
                    sentence_count += 1
                    logger.info(f"📝 Sentence {sentence_count}: '{sentence}'")
                    
                    # Immediately synthesize and send this sentence
                    if sentence and sentence.strip():
                        logger.info(f"🎵 Synthesizing sentence {sentence_count}: '{sentence}'")
                        
                        # Generate TTS for this sentence WITH NATIVE TIMING
                        tts_task = asyncio.create_task(
                            tts_processor.synthesize_initial_speech_with_timing(sentence)
                        )
                        manager.set_task(client_id, "tts", tts_task)

                        # Wait for TTS to complete
                        tts_result = await tts_task
                        if isinstance(tts_result, tuple) and len(tts_result) == 2:
                            audio, timings = tts_result
                        else:
                            audio = tts_result
                            timings = []
                            logger.warning(
                                "TTS returned single value instead of tuple - no timing data available"
                            )

                        logger.info(
                            f"✅ Sentence {sentence_count} TTS complete: {len(audio) if audio is not None else 0} samples, {len(timings)} word timings"
                        )

                        if audio is not None and len(audio) > 0:
                            # Convert to base64 and send to client WITH TIMING DATA
                            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

                            # Send audio with native timing information
                            audio_message = {
                                "audio": base64_audio,
                                "word_timings": timings,
                                "sample_rate": 24000,
                                "method": "native_kokoro_timing",
                                "modality": "multimodal" if image_data else "audio_only",
                                "sentence_number": sentence_count,
                                "streaming": True,
                            }

                            await websocket.send_text(json.dumps(audio_message))
                            logger.info(
                                f"🎤 Sentence {sentence_count} audio sent to client with {len(timings)} NATIVE word timings [{audio_message['modality']}]"
                            )

                # Signal end of audio stream
                await websocket.send_text(json.dumps({"audio_complete": True}))
                logger.info(f"✨ Streaming complete! Sent {sentence_count} sentences")

            except asyncio.CancelledError:
                logger.info("Audio processing cancelled")
                raise
            except Exception as e:
                logger.error(f"Error processing audio segment: {e}")
                # Add more detailed error info
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")

        async def receive_and_process():
            """Receive and process messages from the client"""
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        message = json.loads(data)

                        # Handle interrupt signal
                        if "interrupt" in message and message["interrupt"]:
                            logger.info(f"🛑 Received INTERRUPT signal from {client_id}")
                            await manager.cancel_current_tasks(client_id)
                            continue

                        # Handle complete audio segments from frontend
                        if "audio_segment" in message:
                            # Cancel any current processing
                            await manager.cancel_current_tasks(client_id)

                            # Decode audio data
                            audio_data = base64.b64decode(message["audio_segment"])

                            # Check if image is also included
                            image_data = None
                            if "image" in message:
                                image_data = base64.b64decode(message["image"])
                                logger.info(
                                    f"Received audio+image: audio={len(audio_data)} bytes, image={len(image_data)} bytes"
                                )
                            else:
                                logger.info(
                                    f"Received audio-only: {len(audio_data)} bytes"
                                )

                            # Start processing the audio segment with optional image
                            processing_task = asyncio.create_task(
                                process_audio_segment(audio_data, image_data)
                            )
                            manager.set_task(client_id, "processing", processing_task)

                        # Handle standalone images (only if not currently processing)
                        elif "image" in message:
                            if not (
                                client_id in manager.current_tasks
                                and manager.current_tasks[client_id]["processing"]
                                and not manager.current_tasks[client_id][
                                    "processing"
                                ].done()
                            ):
                                image_data = base64.b64decode(message["image"])
                                manager.update_stats("images_received")

                                # Save standalone image
                                saved_path = manager.image_manager.save_image(
                                    image_data, client_id, "standalone"
                                )
                                if saved_path:
                                    verification = manager.image_manager.verify_image(
                                        saved_path
                                    )
                                    logger.info(
                                        f"📸 Standalone image saved and verified: {verification}"
                                    )

                                await smolvlm_processor.set_image(image_data)
                                logger.info("Image updated")

                        # Handle realtime input (for backward compatibility)
                        elif "realtime_input" in message:
                            for chunk in message["realtime_input"]["media_chunks"]:
                                if chunk["mime_type"] == "audio/pcm":
                                    # Treat as complete audio segment
                                    await manager.cancel_current_tasks(client_id)

                                    audio_data = base64.b64decode(chunk["data"])
                                    processing_task = asyncio.create_task(
                                        process_audio_segment(audio_data)
                                    )
                                    manager.set_task(
                                        client_id, "processing", processing_task
                                    )

                                elif chunk["mime_type"] == "image/jpeg":
                                    # Only process image if not currently processing audio
                                    if not (
                                        client_id in manager.current_tasks
                                        and manager.current_tasks[client_id][
                                            "processing"
                                        ]
                                        and not manager.current_tasks[client_id][
                                            "processing"
                                        ].done()
                                    ):
                                        image_data = base64.b64decode(chunk["data"])
                                        manager.update_stats("images_received")

                                        # Save image from realtime input
                                        saved_path = manager.image_manager.save_image(
                                            image_data, client_id, "realtime"
                                        )
                                        if saved_path:
                                            verification = (
                                                manager.image_manager.verify_image(
                                                    saved_path
                                                )
                                            )
                                            logger.info(
                                                f"📸 Realtime image saved and verified: {verification}"
                                            )

                                        await smolvlm_processor.set_image(image_data)

                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON: {e}")
                        await websocket.send_text(
                            json.dumps({"error": "Invalid JSON format"})
                        )
                    except KeyError as e:
                        logger.error(f"Missing key in message: {e}")
                        await websocket.send_text(
                            json.dumps({"error": f"Missing required field: {e}"})
                        )
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        await websocket.send_text(
                            json.dumps({"error": f"Processing error: {str(e)}"})
                        )

            except WebSocketDisconnect:
                logger.info("WebSocket connection closed during receive loop")

        # Run tasks concurrently
        receive_task = asyncio.create_task(receive_and_process())
        keepalive_task = asyncio.create_task(send_keepalive())

        # Wait for any task to complete (usually due to disconnection or error)
        done, pending = await asyncio.wait(
            [receive_task, keepalive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Log results of completed tasks
        for task in done:
            try:
                result = task.result()
            except Exception as e:
                logger.error(f"Task finished with error: {e}")

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket session error for client {client_id}: {e}")
    finally:
        # Cleanup
        logger.info(f"Cleaning up resources for client {client_id}")
        await manager.cancel_current_tasks(client_id)
        manager.disconnect(client_id)


def main():
    """Main function to start the FastAPI server"""
    logger.info("Starting FastAPI Whisper + SmolVLM2 Voice Assistant server...")

    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        ws_ping_interval=20,
        ws_ping_timeout=60,
        timeout_keep_alive=30,
    )

    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


if __name__ == "__main__":
    main()
