"""Unit tests for the /v1/audio/speech streaming gate.

Uses stdlib unittest — the repo doesn't have a pytest setup and this test
only exercises a pure decision function, so there's no reason to pull one
in. Run with:

    python -m unittest tests.test_openai_streaming
"""

import unittest

from api.routers.streaming_policy import NON_STREAMABLE_FORMATS, should_stream as _should_stream


class ShouldStreamTests(unittest.TestCase):
    def test_default_mp3_streams(self):
        self.assertTrue(_should_stream(speed=1.0, response_format="mp3"))
        self.assertTrue(_should_stream(speed=None, response_format="mp3"))

    def test_flac_streams(self):
        # FLAC frames concatenate cleanly.
        self.assertTrue(_should_stream(speed=1.0, response_format="flac"))

    def test_aac_streams(self):
        # pydub exports AAC as ADTS frames, which concatenate.
        self.assertTrue(_should_stream(speed=1.0, response_format="aac"))

    def test_wav_never_streams(self):
        # WAV header embeds total-length up front.
        self.assertFalse(_should_stream(speed=1.0, response_format="wav"))
        self.assertFalse(_should_stream(speed=None, response_format="wav"))

    def test_m4a_never_streams(self):
        # Per-chunk MP4 containers don't concatenate into valid MP4.
        self.assertFalse(_should_stream(speed=1.0, response_format="m4a"))

    def test_opus_never_streams(self):
        # Per-chunk Ogg produces a chained stream; unreliable on iOS.
        self.assertFalse(_should_stream(speed=1.0, response_format="opus"))

    def test_pcm_never_streams(self):
        # Raw PCM chunks concatenate correctly but no container framing
        # means downstream would silently behave differently; conservative.
        self.assertFalse(_should_stream(speed=1.0, response_format="pcm"))

    def test_non_default_speed_never_streams(self):
        # Speed != 1.0 requires the full numpy array for apply_speed.
        self.assertFalse(_should_stream(speed=0.5, response_format="mp3"))
        self.assertFalse(_should_stream(speed=1.5, response_format="mp3"))
        self.assertFalse(_should_stream(speed=2.0, response_format="flac"))

    def test_speed_none_is_default(self):
        # Missing speed means no speed adjustment, i.e. streamable.
        self.assertTrue(_should_stream(speed=None, response_format="mp3"))

    def test_exclude_set_stays_in_sync(self):
        # Guardrail against drift between the exclude set and the decision
        # function — every format in NON_STREAMABLE_FORMATS must be rejected
        # at default speed, and at least one common streamable format must
        # be accepted.
        for fmt in NON_STREAMABLE_FORMATS:
            self.assertFalse(
                _should_stream(speed=1.0, response_format=fmt),
                f"format {fmt!r} should be rejected",
            )
        self.assertTrue(_should_stream(speed=1.0, response_format="mp3"))


if __name__ == "__main__":
    unittest.main()
