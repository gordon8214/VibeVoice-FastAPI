"""OpenAI-compatible TTS endpoint."""

import asyncio
import logging
import time
from typing import AsyncIterator
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response

from api.models import OpenAITTSRequest, ErrorResponse
from api.routers.streaming_policy import should_stream
from api.services.tts_service import TTSService
from api.services.voice_manager import VoiceManager
from api.utils.audio_utils import audio_to_bytes, get_content_type, get_audio_duration, apply_speed
from api.utils.streaming import create_streaming_response
from api.config import settings

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/v1/audio", tags=["OpenAI Compatible"])


async def _observe_stream(
    inner: AsyncIterator[bytes],
    *,
    log_prefix: str,
) -> AsyncIterator[bytes]:
    """Wrap a streaming body iterator to log post-stream metrics.

    Captures ok / cancelled / error outcomes so streaming responses are
    just as ops-visible as the non-streaming path used to be. Clean-up of
    the underlying generation thread is handled in tts_service._generate_streaming
    via its try/finally; this wrapper only records what happened on the
    HTTP side.
    """
    start = time.monotonic()
    total_bytes = 0
    status = "ok"
    try:
        async for chunk in inner:
            total_bytes += len(chunk)
            yield chunk
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except Exception:
        status = "error"
        raise
    finally:
        elapsed = time.monotonic() - start
        logger.info(
            f"{log_prefix} | Status: {status} | Bytes: {total_bytes} | Elapsed: {elapsed:.2f}s"
        )


# Global service instances (initialized in main.py)
tts_service: TTSService = None
voice_manager: VoiceManager = None


def get_tts_service() -> TTSService:
    """Dependency to get TTS service, triggering lazy load if needed."""
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not ready")
    tts_service.ensure_loaded()
    return tts_service


def get_voice_manager() -> VoiceManager:
    """Dependency to get voice manager."""
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="Voice manager not initialized")
    return voice_manager


@router.post("/speech")
async def create_speech(
    request: OpenAITTSRequest,
    tts: TTSService = Depends(get_tts_service),
    voices: VoiceManager = Depends(get_voice_manager)
):
    """
    Generate speech from text using OpenAI-compatible API.
    
    This endpoint mimics the OpenAI TTS API for compatibility with existing clients.
    """
    try:
        # Try loading as OpenAI voice first, then as direct VibeVoice preset
        voice_audio = voices.load_voice_audio(request.voice, is_openai_voice=True)
        
        # If not found as OpenAI voice, try as direct VibeVoice preset name
        if voice_audio is None:
            voice_audio = voices.load_voice_audio(request.voice, is_openai_voice=False)
        
        if voice_audio is None:
            available_openai = ', '.join(voices.OPENAI_VOICE_MAPPING.keys())
            available_presets = ', '.join(sorted(voices.voice_presets.keys()))
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{request.voice}' not found. OpenAI voices: {available_openai}. VibeVoice presets: {available_presets}"
            )
        
        # Format text as single-speaker script
        formatted_script = tts.format_script_for_single_speaker(request.input, speaker_id=0)

        wants_speed = request.speed is not None and request.speed != 1.0
        text_preview = request.input[:100] + "..." if len(request.input) > 100 else request.input

        if should_stream(request.speed, request.response_format):
            logger.info(
                f"Generating speech (streaming) - Text: {text_preview} | Voice: {request.voice} | "
                f"Model: {request.model} ({settings.vibevoice_model_path}) | "
                f"CFG: {settings.default_cfg_scale} | Format: {request.response_format}"
            )
            audio_stream = tts.generate_speech(
                text=formatted_script,
                voice_samples=[voice_audio],
                cfg_scale=settings.default_cfg_scale,
                stream=True,
            )
            streaming_response = create_streaming_response(
                audio_stream,
                format=request.response_format,
                sample_rate=24000,
                use_sse=False,
            )
            # Rewrap the body iterator with a metrics wrapper so we still log
            # bytes / elapsed / status after a streaming response finishes or
            # is cancelled. Thread-side cleanup lives in
            # tts_service._generate_streaming's try/finally.
            log_prefix = (
                f"Generated speech (streaming) - Text: {text_preview} | Voice: {request.voice} | "
                f"Model: {request.model} ({settings.vibevoice_model_path}) | "
                f"CFG: {settings.default_cfg_scale} | Format: {request.response_format}"
            )
            streaming_response.body_iterator = _observe_stream(
                streaming_response.body_iterator,
                log_prefix=log_prefix,
            )
            return streaming_response

        # Non-streaming path: speed adjustment or a format we can't chunk.
        start_time = time.time()
        audio = tts.generate_speech(
            text=formatted_script,
            voice_samples=[voice_audio],
            cfg_scale=settings.default_cfg_scale,
            stream=False,
        )
        generation_time = time.time() - start_time

        if wants_speed:
            audio = apply_speed(audio, request.speed, sample_rate=24000)

        audio_duration = get_audio_duration(audio, sample_rate=24000)

        speed_info = f" | Speed: {request.speed}" if wants_speed else ""
        logger.info(
            f"Generated speech - Text: {text_preview} | Voice: {request.voice} | "
            f"Model: {request.model} ({settings.vibevoice_model_path}) | "
            f"CFG: {settings.default_cfg_scale}{speed_info} | Audio Duration: {audio_duration:.2f}s | Generation Time: {generation_time:.2f}s"
        )

        audio_bytes = audio_to_bytes(
            audio,
            sample_rate=24000,
            format=request.response_format,
        )

        return Response(
            content=audio_bytes,
            media_type=get_content_type(request.response_format),
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices")
async def list_voices(
    voices: VoiceManager = Depends(get_voice_manager)
):
    """
    List all available voices in OpenAI-compatible format.
    
    Returns OpenAI standard voices (if their mapped presets exist) plus all custom voices.
    """
    try:
        voice_list = []
        
        # Add OpenAI standard voices (if their mapped presets exist)
        for openai_name, vibevoice_preset in voices.OPENAI_VOICE_MAPPING.items():
            if vibevoice_preset in voices.voice_presets:
                voice_list.append({
                    "id": openai_name,
                    "object": "voice",
                    "name": openai_name
                })
        
        # Add all custom voices from VOICES_DIR
        all_voices = voices.list_available_voices()
        for voice in all_voices:
            # Skip if already added as OpenAI voice
            if voice["name"] not in voices.OPENAI_VOICE_MAPPING.values():
                voice_list.append({
                    "id": voice["name"],
                    "object": "voice",
                    "name": voice["name"]
                })
        
        return {
            "object": "list",
            "data": voice_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

