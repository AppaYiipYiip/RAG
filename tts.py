# tts.py
import io
import logging
import asyncio
import edge_tts

logger = logging.getLogger(__name__)

# Voice for French (you can change the voice name)
VOICE = "fr-FR-HenriNeural"   # Male French voice; alternatives: "fr-FR-DeniseNeural" (female)

async def _synthesize_async(text: str) -> bytes:
    """Generate speech audio (MP3) from text using edge-tts."""
    communicate = edge_tts.Communicate(text, VOICE)
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    return audio_bytes

def synthesize_speech(text: str) -> bytes:
    """
    Convert French text to speech and return audio bytes (MP3).
    This function blocks until synthesis is complete.
    """
    logger.info(f"Synthesizing speech for text: {text[:80]}...")
    try:
        # Run the async function in a synchronous context
        audio = asyncio.run(_synthesize_async(text))
        logger.info(f"Generated {len(audio)} bytes of audio.")
        return audio
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise