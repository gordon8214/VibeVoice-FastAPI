"""Audio format conversion and processing utilities."""

import io
import numpy as np
import torch
import librosa
from typing import Union, Literal
from pydub import AudioSegment
import soundfile as sf


AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "m4a"]


def apply_speed(
    audio: Union[np.ndarray, torch.Tensor],
    speed: float,
    sample_rate: int = 24000
) -> np.ndarray:
    """
    Apply speed change to audio using time-stretching.

    Args:
        audio: Audio data as numpy array or torch tensor
        speed: Speed factor (0.25-4.0). <1.0 = slower, >1.0 = faster
        sample_rate: Sample rate of the audio

    Returns:
        Time-stretched audio as numpy array
    """
    if speed == 1.0:
        if torch.is_tensor(audio):
            return audio.detach().cpu().numpy()
        return np.array(audio, dtype=np.float32)

    # Convert tensor to numpy if needed
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().numpy()

    audio = np.array(audio, dtype=np.float32)

    # Ensure 1D
    if len(audio.shape) > 1:
        audio = audio.squeeze()

    # Use librosa time_stretch (preserves pitch while changing speed)
    # speed > 1.0 = faster (shorter), speed < 1.0 = slower (longer)
    stretched = librosa.effects.time_stretch(audio, rate=speed)

    return stretched.astype(np.float32)


def convert_to_16_bit_wav(audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert audio to 16-bit PCM format.
    
    Args:
        audio: Audio data as numpy array or torch tensor
        
    Returns:
        16-bit PCM audio as numpy array
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().numpy()
    
    # Ensure numpy array
    audio = np.array(audio, dtype=np.float32)
    
    # Ensure 1D
    if len(audio.shape) > 1:
        audio = audio.squeeze()
    
    # Normalize to range [-1, 1] if needed
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
    
    # Scale to 16-bit integer range
    audio_16bit = (audio * 32767).astype(np.int16)
    
    return audio_16bit


def audio_to_bytes(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 24000,
    format: AudioFormat = "mp3",
    bitrate: str = "128k"
) -> bytes:
    """
    Convert audio array to bytes in specified format.
    
    Args:
        audio: Audio data as numpy array or torch tensor
        sample_rate: Sample rate of the audio
        format: Output format (mp3, opus, aac, flac, wav, pcm)
        bitrate: Bitrate for lossy formats (e.g., "128k", "192k")
        
    Returns:
        Audio data as bytes
    """
    # Convert to 16-bit PCM
    audio_16bit = convert_to_16_bit_wav(audio)
    
    if format == "pcm":
        # Return raw PCM data
        return audio_16bit.tobytes()
    
    elif format == "wav":
        # Create WAV file in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio_16bit, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()
    
    else:
        # Use pydub for other formats (mp3, opus, aac, flac)
        # First create WAV in memory
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_16bit, sample_rate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        
        # Load with pydub
        audio_segment = AudioSegment.from_wav(wav_buffer)
        
        # Export to target format
        output_buffer = io.BytesIO()
        
        export_params = {
            "format": format,
        }
        
        # Add bitrate for lossy formats
        if format in ["mp3", "opus", "aac", "m4a"]:
            export_params["bitrate"] = bitrate
        
        # Special handling for opus
        if format == "opus":
            export_params["codec"] = "libopus"
        
        # m4a is AAC in MP4 container
        if format == "m4a":
            export_params["codec"] = "aac"
        
        audio_segment.export(output_buffer, **export_params)
        output_buffer.seek(0)
        
        return output_buffer.read()


def get_audio_duration(audio: Union[np.ndarray, torch.Tensor], sample_rate: int = 24000) -> float:
    """
    Get duration of audio in seconds.
    
    Args:
        audio: Audio data as numpy array or torch tensor
        sample_rate: Sample rate of the audio
        
    Returns:
        Duration in seconds
    """
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().numpy()
    
    audio = np.array(audio)
    if len(audio.shape) > 1:
        audio = audio.squeeze()
    
    return len(audio) / sample_rate


def get_content_type(format: AudioFormat) -> str:
    """
    Get MIME content type for audio format.
    
    Args:
        format: Audio format
        
    Returns:
        MIME content type string
    """
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "application/octet-stream",
        "m4a": "audio/mp4"
    }
    return content_types.get(format, "application/octet-stream")


def concatenate_audio_chunks(
    chunks: list[Union[np.ndarray, torch.Tensor]]
) -> np.ndarray:
    """
    Concatenate multiple audio chunks into a single array.
    
    Args:
        chunks: List of audio chunks
        
    Returns:
        Concatenated audio as numpy array
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    
    # Convert all chunks to numpy
    numpy_chunks = []
    for chunk in chunks:
        if torch.is_tensor(chunk):
            chunk = chunk.detach().cpu().numpy()
        chunk = np.array(chunk, dtype=np.float32)
        if len(chunk.shape) > 1:
            chunk = chunk.squeeze()
        numpy_chunks.append(chunk)
    
    # Concatenate
    return np.concatenate(numpy_chunks)


