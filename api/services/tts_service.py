"""Core TTS generation service wrapping VibeVoice model."""

import threading

import torch
import numpy as np
from typing import Iterator, List, Optional, Union
from transformers import set_seed, BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

from api.config import Settings


# Qwen2.5's tokenizer (the LLM backbone under VibeVoice) maps Unicode
# "smart" punctuation to rarer token IDs than their ASCII equivalents.
# Those rare tokens have much less acoustic training, so contractions like
# "it\u2019s" are pronounced as "it" (the suffix is dropped), and smart
# quotes / NBSP produce degraded audio. Normalizing to ASCII routes the
# text through the well-trained token path.
#
# Left intentionally: U+2014 em-dash (we use it as a paragraph-pause cue
# and the model handles it well).
_UNICODE_PUNCT_NORMALIZATION = str.maketrans({
    "\u2019": "'",   # right single quotation mark / apostrophe (fixes it's/don't/etc.)
    "\u2018": "'",   # left single quotation mark
    "\u201C": '"',   # left double quotation mark
    "\u201D": '"',   # right double quotation mark
    "\u2013": "-",   # en-dash
    "\u00A0": " ",   # non-breaking space
    "\u2026": "...", # horizontal ellipsis
})


def _normalize_text_for_tts(text: str) -> str:
    """Normalize Unicode punctuation to ASCII for better Qwen tokenization."""
    return text.translate(_UNICODE_PUNCT_NORMALIZATION)


class TTSService:
    """Service for TTS generation using VibeVoice model."""

    def __init__(self, settings: Settings):
        """
        Initialize TTS service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None
        self._model_loaded = False
        self._load_lock = threading.Lock()

    def load_model(self):
        """Load VibeVoice model and processor."""
        if self._model_loaded:
            print("Model already loaded")
            return

        print(f"Loading VibeVoice model from {self.settings.vibevoice_model_path}")

        # Get device and dtype
        self.device = self.settings.get_device()
        self.dtype = self.settings.get_dtype()
        attn_implementation = self.settings.get_attn_implementation()

        print(f"Using device: {self.device}, dtype: {self.dtype}, attention: {attn_implementation}")

        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.settings.vibevoice_model_path)

        # Setup bitsandbytes 4-bit quantization config if enabled
        quantization_config = None
        if self.settings.vibevoice_load_in_4bit and self.device == "cuda":
            print("Using 4-bit quantization (reduces VRAM from ~18GB to ~7GB)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )

        # Determine if we should load to CPU first for torchao quantization
        # This avoids loading full precision model to GPU then quantizing (wastes VRAM)
        load_to_cpu_first = (
            self.settings.vibevoice_quantization
            and self.device == "cuda"
        )

        if load_to_cpu_first:
            print("Loading model to CPU first for quantization (saves GPU memory)...")
            # Use sdpa for CPU loading since flash_attention_2 requires CUDA
            cpu_attn = "sdpa" if attn_implementation == "flash_attention_2" else attn_implementation
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.settings.vibevoice_model_path,
                torch_dtype=self.dtype,
                device_map="cpu",
                attn_implementation=cpu_attn,
                low_cpu_mem_usage=True,
            )
            self.model.eval()

            # Apply quantization on CPU
            self._apply_quantization()

            # Now move to CUDA
            print("Moving quantized model to CUDA...")
            self.model = self.model.to("cuda")

            # Log final VRAM usage
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                vram_final = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"Final VRAM usage after moving to GPU: {vram_final:.2f} GB")
        else:
            # Standard loading path (no torchao quantization or non-CUDA device)
            try:
                if self.device == "mps":
                    # MPS doesn't support device_map or quantization
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.settings.vibevoice_model_path,
                        torch_dtype=self.dtype,
                        attn_implementation=attn_implementation,
                        device_map=None,
                    )
                    self.model.to("mps")
                elif self.device == "cuda":
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.settings.vibevoice_model_path,
                        torch_dtype=self.dtype,
                        device_map="cuda",
                        attn_implementation=attn_implementation,
                        quantization_config=quantization_config,
                    )
                else:
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.settings.vibevoice_model_path,
                        torch_dtype=self.dtype,
                        device_map="cpu",
                        attn_implementation=attn_implementation,
                    )
            except Exception as e:
                if attn_implementation == 'flash_attention_2':
                    print(f"Flash attention failed: {e}")
                    print("Falling back to SDPA attention")
                    attn_implementation = "sdpa"

                    if self.device == "mps":
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.settings.vibevoice_model_path,
                            torch_dtype=self.dtype,
                            attn_implementation=attn_implementation,
                            device_map=None,
                        )
                        self.model.to("mps")
                    elif self.device == "cuda":
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.settings.vibevoice_model_path,
                            torch_dtype=self.dtype,
                            device_map="cuda",
                            attn_implementation=attn_implementation,
                            quantization_config=quantization_config,
                        )
                    else:
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.settings.vibevoice_model_path,
                            torch_dtype=self.dtype,
                            device_map="cpu",
                            attn_implementation=attn_implementation,
                        )
                else:
                    raise e

            self.model.eval()

        # Apply torch.compile for optimized inference
        if self.settings.torch_compile:
            try:
                compile_mode = self.settings.torch_compile_mode
                self.model = torch.compile(self.model, mode=compile_mode, dynamic=True)
                print(f"Model compiled with torch.compile(mode='{compile_mode}', dynamic=True)")
            except Exception as e:
                print(f"torch.compile() failed: {e}, continuing without compilation")

        # Configure noise scheduler
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )

        # Set inference steps
        self.model.set_ddpm_inference_steps(num_steps=self.settings.vibevoice_inference_steps)

        self._model_loaded = True
        print("Model loaded successfully")

    def _apply_quantization(self):
        """Apply quantization to the model based on settings."""
        quant_method = self.settings.vibevoice_quantization

        if quant_method == "int8_torchao":
            self._apply_torchao_quant(bits=8)
        elif quant_method == "int4_torchao":
            self._apply_torchao_quant(bits=4)
        else:
            logger.warning(f"Unknown quantization method: {quant_method}, skipping quantization")

    def _apply_torchao_quant(self, bits: int = 8):
        """
        Apply torchao weight-only quantization to the language model.

        This selectively quantizes only the LLM (Qwen2) decoder and lm_head,
        keeping audio components (tokenizers, diffusion head, connectors) at full precision.

        Args:
            bits: 8 for INT8 (~40% VRAM reduction) or 4 for INT4 (~60% VRAM reduction, faster)
        """
        try:
            from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
        except ImportError:
            logger.error(
                "torchao not installed. Install with: pip install torchao\n"
                "Falling back to full precision."
            )
            return

        # Select quantization function based on bits
        if bits == 4:
            quant_fn = int4_weight_only()
            quant_name = "INT4"
        else:
            quant_fn = int8_weight_only()
            quant_name = "INT8"

        # Check if model is on CUDA (for memory logging)
        model_on_cuda = next(self.model.parameters()).is_cuda

        logger.info(f"Applying torchao {quant_name} weight-only quantization...")
        if model_on_cuda:
            logger.info("Model is on CUDA - quantizing in place")
        else:
            logger.info("Model is on CPU - quantizing before moving to GPU (saves VRAM)")

        # Quantize only the language model (Qwen2 decoder) - this is the largest component
        # The audio components (acoustic_tokenizer, semantic_tokenizer, prediction_head, connectors)
        # are kept at full precision to maintain audio quality
        try:
            logger.info("Quantizing language_model (Qwen2 decoder)...")
            quantize_(self.model.model.language_model, quant_fn)

            logger.info("Quantizing lm_head...")
            quantize_(self.model.lm_head, quant_fn)

        except Exception as e:
            logger.error(f"Failed to quantize model: {e}")
            logger.info("Continuing with full precision model")
            return

        logger.info(f"{quant_name} quantization applied successfully")

        # Force garbage collection
        import gc
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def ensure_loaded(self):
        """Ensure model is loaded, loading on first call."""
        if self._model_loaded:
            return
        with self._load_lock:
            if self._model_loaded:
                return
            print("Lazy-loading VibeVoice model on first request...")
            self.load_model()

    def unload_model(self):
        """Unload model from VRAM and free GPU memory."""
        with self._load_lock:
            if not self._model_loaded:
                return
            print("Unloading VibeVoice model...")
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            self._model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("VibeVoice model unloaded, VRAM freed")

    def generate_speech(
        self,
        text: str,
        voice_samples: List[np.ndarray],
        cfg_scale: float = 1.3,
        inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        stream: bool = False
    ) -> Union[np.ndarray, Iterator[np.ndarray]]:
        """
        Generate speech from text.

        Args:
            text: Input text (formatted with Speaker labels)
            voice_samples: List of voice sample arrays
            cfg_scale: Classifier-free guidance scale
            inference_steps: Number of diffusion steps (None = use default)
            seed: Random seed for reproducibility
            stream: Whether to return streaming iterator

        Returns:
            Generated audio array or iterator of audio chunks
        """
        self.ensure_loaded()

        # Normalize Unicode punctuation to ASCII so contractions like
        # "it\u2019s" tokenize via the well-trained path and don't drop
        # suffixes. See _normalize_text_for_tts for full rationale.
        text = _normalize_text_for_tts(text)

        # Set seed if provided
        if seed is not None:
            set_seed(seed)

        # Set inference steps if provided
        if inference_steps is not None:
            self.model.set_ddpm_inference_steps(num_steps=inference_steps)

        # Process inputs
        inputs = self.processor(
            text=[text],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move to device
        target_device = self.device if self.device in ("cuda", "mps") else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device)

        if stream:
            # Return streaming iterator
            return self._generate_streaming(inputs, cfg_scale)
        else:
            # Generate all at once
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={'do_sample': False},
                    return_speech=True,
                    verbose=False,
                    refresh_negative=True,
                    show_progress_bar=False
                )

            # Get audio output
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                audio = outputs.speech_outputs[0]
                if torch.is_tensor(audio):
                    # Convert bfloat16 to float32 before converting to numpy
                    if audio.dtype == torch.bfloat16:
                        audio = audio.float()
                    audio = audio.cpu().numpy()
                return audio
            else:
                raise RuntimeError("No audio generated")

    def _generate_streaming(
        self,
        inputs: dict,
        cfg_scale: float
    ) -> Iterator[np.ndarray]:
        """
        Generate speech with streaming.

        Args:
            inputs: Processed model inputs
            cfg_scale: CFG scale

        Yields:
            Audio chunks as numpy arrays
        """
        # Create audio streamer
        audio_streamer = AudioStreamer(
            batch_size=1,
            stop_signal=None,
            timeout=None
        )

        # Start generation in background
        import threading

        def generate():
            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={'do_sample': False},
                    audio_streamer=audio_streamer,
                    return_speech=True,
                    verbose=False,
                    refresh_negative=True,
                    show_progress_bar=False
                )

        generation_thread = threading.Thread(target=generate)
        generation_thread.start()

        # Yield chunks as they arrive
        audio_stream = audio_streamer.get_stream(0)
        for chunk in audio_stream:
            if torch.is_tensor(chunk):
                # Convert bfloat16 to float32 before converting to numpy
                if chunk.dtype == torch.bfloat16:
                    chunk = chunk.float()
                chunk = chunk.cpu().numpy()
            yield chunk

        # Wait for generation to complete
        generation_thread.join(timeout=10.0)

    def format_script_for_single_speaker(self, text: str, speaker_id: int = 0) -> str:
        """
        Format plain text as single-speaker script.

        Collapses the input into a single "Speaker {id}:" turn, because
        multiple turns (one per line/paragraph) cause cloned voices to
        emit emergent transition sounds — grinding, music stings, etc. —
        at every turn boundary.

        To preserve prosodic pauses at paragraph/heading boundaries, each
        non-empty input line is:
          1. Ensured to end with sentence-ending punctuation (period added
             if the line is a bare heading like "Key context"), and
          2. Joined with " — " (spaced em-dash). The em-dash gives the
             model's LLM head a strong pause cue, equivalent to the
             paragraph break in the source text, without creating a new
             dialogue turn that would trigger ambient-audio artifacts.

        Embedded newlines within a turn truncate generation (the model
        stops at the first \\n), so the output is a single continuous
        line.

        Args:
            text: Plain text input
            speaker_id: Speaker ID to use

        Returns:
            Formatted script (one "Speaker {id}:" turn, em-dash-separated)
        """
        sentence_end = {".", "!", "?", ":", ";", "—", "…"}
        processed = []
        for raw in text.strip().split("\n"):
            line = raw.strip()
            if not line:
                continue
            if line[-1] not in sentence_end:
                line = line + "."
            processed.append(line)
        body = " — ".join(processed)
        return f"Speaker {speaker_id}: {body}"
