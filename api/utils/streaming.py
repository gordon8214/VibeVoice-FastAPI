"""Streaming utilities for real-time audio delivery."""

import asyncio
import json
import logging
from io import BytesIO
from typing import AsyncIterator, Iterator, Optional, Union
from fastapi.responses import StreamingResponse
import av
import numpy as np
import torch

logger = logging.getLogger(__name__)


# PyAV codec map for the persistent-encoder streaming path. Formats not in
# this map use per-chunk encoding (or aren't streamable at all).
_AV_CODEC_MAP = {
    "mp3": "mp3",
    "opus": "libopus",
    "flac": "flac",
    "aac": "aac",
}


class _PersistentEncoder:
    """Wraps a single PyAV container + encoder so all chunks of a streaming
    response share one encoder instance. Without this, each chunk gets its
    own LAME/Opus encoder with its own priming silence (~48 ms for MP3 at
    24 kHz) producing audible gaps between every chunk — which was the
    audio-quality regression reported against the chunked-MP3 streaming
    path.
    """

    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        self.format = format
        self.sample_rate = sample_rate
        self.channels = channels
        self.pts = 0
        self.output_buffer = BytesIO()

        container_options: dict = {}
        # Disabling the Xing VBR header produces an MP3 without a stored
        # duration, which is what we want: an incrementally delivered file
        # where the true length isn't known until the end.
        if format == "mp3":
            container_options = {"write_xing": "0"}

        # AAC goes in ADTS (framed streaming), not MP4.
        container_format = "adts" if format == "aac" else format
        self.container = av.open(
            self.output_buffer,
            mode="w",
            format=container_format,
            options=container_options,
        )
        self.stream = self.container.add_stream(
            _AV_CODEC_MAP[format],
            rate=sample_rate,
            layout="mono" if channels == 1 else "stereo",
        )
        if format in {"mp3", "aac", "opus"}:
            self.stream.bit_rate = 128_000

    def write(self, chunk: np.ndarray) -> bytes:
        """Feed one audio chunk and return whatever bytes the encoder emits."""
        if chunk.size == 0:
            return b""

        # The model yields float32 in [-1, 1]; PyAV wants int16 PCM here.
        if chunk.dtype != np.int16:
            chunk = np.clip(chunk, -1.0, 1.0)
            chunk = (chunk * 32767.0).astype(np.int16)

        frame = av.AudioFrame.from_ndarray(
            chunk.reshape(1, -1),
            format="s16",
            layout="mono" if self.channels == 1 else "stereo",
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self.pts
        self.pts += frame.samples

        for packet in self.stream.encode(frame):
            self.container.mux(packet)

        return self._drain_buffer()

    def finalize(self) -> bytes:
        """Flush the encoder and close the container. Idempotent."""
        if self.container is None:
            return b""
        try:
            for packet in self.stream.encode(None):
                self.container.mux(packet)
        finally:
            self.container.close()
            self.container = None  # mark done so repeat finalize is a no-op
        tail = self._drain_buffer()
        self.output_buffer.close()
        return tail

    def _drain_buffer(self) -> bytes:
        data = self.output_buffer.getvalue()
        self.output_buffer.seek(0)
        self.output_buffer.truncate(0)
        return data


async def audio_chunk_generator(
    audio_stream: Iterator,
    format: str = "mp3",
    sample_rate: int = 24000,
) -> AsyncIterator[bytes]:
    """Generate audio chunks for streaming response.

    For codecs that tolerate stateful encoding (mp3/opus/flac/aac) we keep
    a single PyAV encoder open across the whole response so there's no
    per-chunk priming gap. PCM bypasses the encoder entirely. Any other
    format falls back to per-chunk pydub encoding, which is byte-correct
    for simple formats but may produce audible gaps — callers should
    exclude those formats via the streaming-policy check upstream.
    """
    if format == "pcm":
        async for chunk in _pcm_chunks(audio_stream):
            yield chunk
        return

    if format in _AV_CODEC_MAP:
        async for chunk in _encoded_chunks(audio_stream, format, sample_rate):
            yield chunk
        return

    # Unsupported-in-streaming format: fall back to per-chunk pydub. This
    # path exists only so legacy callers don't crash; the openai_tts
    # streaming policy blocks formats that would sound wrong here.
    from api.utils.audio_utils import audio_to_bytes

    for chunk in audio_stream:
        yield audio_to_bytes(chunk, sample_rate=sample_rate, format=format)
        await asyncio.sleep(0)


async def _pcm_chunks(audio_stream: Iterator) -> AsyncIterator[bytes]:
    for chunk in audio_stream:
        if chunk is None or chunk.size == 0:
            continue
        if chunk.dtype != np.int16:
            chunk = np.clip(chunk, -1.0, 1.0)
            chunk = (chunk * 32767.0).astype(np.int16)
        yield chunk.tobytes()
        await asyncio.sleep(0)


async def _encoded_chunks(
    audio_stream: Iterator,
    format: str,
    sample_rate: int,
) -> AsyncIterator[bytes]:
    encoder = _PersistentEncoder(format=format, sample_rate=sample_rate)
    finalized = False
    try:
        for chunk in audio_stream:
            if chunk is None:
                continue
            data = encoder.write(chunk)
            if data:
                yield data
            await asyncio.sleep(0)
        # Normal completion: flush and yield the final trailer bytes.
        tail = encoder.finalize()
        finalized = True
        if tail:
            yield tail
    finally:
        if not finalized:
            # Error or cancellation mid-stream. Still flush so the container
            # isn't left with a half-written output buffer, and so PyAV
            # releases its libav handles. We discard any trailer bytes —
            # the consumer already saw an error / disconnect.
            try:
                encoder.finalize()
            except Exception:
                logger.exception("Persistent encoder finalize failed during cleanup")


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


