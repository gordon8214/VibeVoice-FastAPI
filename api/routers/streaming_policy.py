"""Pure streaming-decision logic for /v1/audio/speech.

Kept out of openai_tts.py so unit tests can import it without pulling in
torch / transformers / vibevoice via the TTS service module.
"""

from typing import Optional


# Formats that cannot be produced correctly as a concatenated byte stream
# of independent-per-chunk encodes.
#   - wav: total-length header up front; per-chunk WAV files can't be joined.
#   - m4a: each chunk is a complete MP4 container (moov atom before mdat);
#          concatenating yields invalid MP4 that most decoders reject.
#   - opus: pydub wraps opus in Ogg; per-chunk Ogg streams produce a chained
#           Ogg file that many players (including iOS AVFoundation) handle
#           unreliably. Revisit if we round-trip a real chained-Ogg test.
#   - pcm: raw PCM concatenates correctly, but the OpenAI-style consumer
#          expects a container-framed file here; leaving out to avoid a
#          silent behavior change.
NON_STREAMABLE_FORMATS = frozenset({"wav", "m4a", "opus", "pcm"})


def should_stream(speed: Optional[float], response_format: str) -> bool:
    """Decide whether /v1/audio/speech can stream the response.

    Streaming requires that we never need the complete audio tensor on the
    server (speed adjustment rescales it) and that the chosen container
    format survives being delivered as independent per-chunk encodes.
    """
    wants_speed = speed is not None and speed != 1.0
    return not wants_speed and response_format not in NON_STREAMABLE_FORMATS
