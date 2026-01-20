"""Core TTS generation service wrapping VibeVoice model."""

import torch
import numpy as np
from typing import Iterator, List, Optional, Union
from transformers import set_seed

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

from api.config import Settings


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
        
        # Check if using quantized model loading
        use_quantized = self.settings.vibevoice_use_quantized
        
        if use_quantized:
            print("Using quantized model loading (AutoModelForCausalLM)")
            self._load_quantized_model()
        else:
            print("Using full precision model loading (VibeVoiceForConditionalGenerationInference)")
            print(f"Using device: {self.device}, dtype: {self.dtype}, attention: {attn_implementation}")
            self._load_full_precision_model(attn_implementation)
        
        self._model_loaded = True
        print("Model loaded successfully")
    
    def _load_quantized_model(self):
        """Load model using AutoModelForCausalLM for quantized models."""
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.settings.vibevoice_model_path,
            trust_remote_code=True
        )
        
        # Load model with quantized settings
        if self.device == "mps":
            # MPS doesn't support device_map="auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.vibevoice_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=None,
            )
            self.model.to("mps")
        else:
            # Use device_map="auto" for CUDA/CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.vibevoice_model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        
        self.model.eval()
        
        # Configure noise scheduler
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        
        # Set inference steps
        self.model.set_ddpm_inference_steps(num_steps=self.settings.vibevoice_inference_steps)
    
    def _load_full_precision_model(self, attn_implementation: str):
        """Load model using full precision VibeVoiceForConditionalGenerationInference."""
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.settings.vibevoice_model_path)
        
        # Load model
        try:
            if self.device == "mps":
                # MPS doesn't support device_map
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
        
        # Configure noise scheduler
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        
        # Set inference steps
        self.model.set_ddpm_inference_steps(num_steps=self.settings.vibevoice_inference_steps)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
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
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
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
        
        Args:
            text: Plain text input
            speaker_id: Speaker ID to use
            
        Returns:
            Formatted script
        """
        # Split into sentences/paragraphs
        lines = text.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                formatted_lines.append(f"Speaker {speaker_id}: {line}")
        
        return '\n'.join(formatted_lines)

