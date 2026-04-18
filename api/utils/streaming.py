"""Streaming utilities for real-time audio delivery."""

import asyncio
import json
from typing import AsyncIterator, Iterator, Union
from fastapi.responses import StreamingResponse
import numpy as np
import torch


async def audio_chunk_generator(
    audio_stream: Iterator,
    format: str = "mp3",
    sample_rate: int = 24000
) -> AsyncIterator[bytes]:
    """
    Generate audio chunks for streaming response.
    
    Args:
        audio_stream: Iterator yielding audio chunks
        format: Audio format for encoding
        sample_rate: Sample rate of audio
        
    Yields:
        Encoded audio chunk bytes
    """
    from api.utils.audio_utils import audio_to_bytes
    
    for chunk in audio_stream:
        # Convert chunk to bytes in target format
        chunk_bytes = audio_to_bytes(chunk, sample_rate=sample_rate, format=format)
        yield chunk_bytes
        
        # Allow other tasks to run
        await asyncio.sleep(0)


async def sse_audio_generator(
    audio_stream: Iterator,
    format: str = "mp3",
    sample_rate: int = 24000
) -> AsyncIterator[str]:
    """
    Generate Server-Sent Events for audio streaming.
    
    Args:
        audio_stream: Iterator yielding audio chunks
        format: Audio format for encoding
        sample_rate: Sample rate of audio
        
    Yields:
        SSE-formatted messages
    """
    from api.utils.audio_utils import audio_to_bytes
    import base64
    
    chunk_id = 0
    
    try:
        for chunk in audio_stream:
            # Convert chunk to bytes
            chunk_bytes = audio_to_bytes(chunk, sample_rate=sample_rate, format=format)
            
            # Encode as base64 for SSE
            chunk_base64 = base64.b64encode(chunk_bytes).decode('utf-8')
            
            # Create SSE message
            event_data = {
                "chunk_id": chunk_id,
                "audio": chunk_base64,
                "format": format,
                "sample_rate": sample_rate
            }
            
            yield f"data: {json.dumps(event_data)}\n\n"
            
            chunk_id += 1
            await asyncio.sleep(0)
        
        # Send completion event
        yield f"data: {json.dumps({'done': True})}\n\n"
        
    except Exception as e:
        # Send error event
        error_data = {
            "error": str(e),
            "type": type(e).__name__
        }
        yield f"data: {json.dumps(error_data)}\n\n"


def create_streaming_response(
    audio_stream: Iterator,
    format: str = "mp3",
    sample_rate: int = 24000,
    use_sse: bool = False
) -> StreamingResponse:
    """
    Create a FastAPI StreamingResponse for audio.
    
    Args:
        audio_stream: Iterator yielding audio chunks
        format: Audio format
        sample_rate: Sample rate
        use_sse: Whether to use Server-Sent Events format
        
    Returns:
        FastAPI StreamingResponse
    """
    from api.utils.audio_utils import get_content_type
    
    if use_sse:
        return StreamingResponse(
            sse_audio_generator(audio_stream, format, sample_rate),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        return StreamingResponse(
            audio_chunk_generator(audio_stream, format, sample_rate),
            media_type=get_content_type(format),
            headers={
                "Transfer-Encoding": "chunked",
                "Cache-Control": "no-cache"
            }
        )


