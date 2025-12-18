import asyncio
import logging
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoFeatureExtractor
import torch
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class EmotionProcessor:
    """Lightweight emotion detection from audio using SenseVoice-style approach"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize emotion detection - runs in background, zero latency impact"""
        self.model = None
        self.feature_extractor = None
        self.device = None
        self.enabled = False
        
        try:
            logger.info("Initializing Emotion Detection (Background)...")
            
            # Use Wav2Vec2 for emotion classification (lighter than SenseVoice)
            # This is a fallback - you can replace with SenseVoiceSmall if available
            from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
            
            model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading emotion model on {self.device}...")
            
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            # Emotion labels from this model
            self.emotion_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
            
            self.enabled = True
            logger.info("✅ Emotion Detection ready (Background mode)")
            
        except Exception as e:
            logger.warning(f"⚠️ Emotion detection unavailable: {e}")
            self.enabled = False
    
    async def detect_emotion_async(self, audio_data: bytes) -> Optional[Dict[str, any]]:
        """
        Detect emotion from audio - runs in background, never blocks main pipeline
        Returns: {"emotion": "happy", "confidence": 0.87} or None if not ready/failed
        """
        if not self.enabled:
            return None
        
        try:
            # Convert audio bytes to numpy array
            # Assuming 16kHz, 16-bit PCM (same as Whisper expects)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._detect_sync, audio_np)
            
            return result
            
        except Exception as e:
            logger.debug(f"Emotion detection error (non-critical): {e}")
            return None
    
    def _detect_sync(self, audio_np: np.ndarray) -> Dict[str, any]:
        """Synchronous emotion detection - runs in thread pool"""
        # Extract features
        inputs = self.feature_extractor(
            audio_np, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get prediction
        predicted_id = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1)[0][predicted_id].item()
        
        emotion = self.emotion_labels[predicted_id]
        
        logger.info(f"🎭 Detected emotion: {emotion} ({confidence:.2f})")
        
        return {
            "emotion": emotion,
            "confidence": confidence
        }
