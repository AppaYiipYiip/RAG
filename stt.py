# stt.py
import io
import logging
import torch
import torchaudio
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from config import WHISPER_MODEL, MAX_AUDIO_DURATION_SECONDS

# Set up module-level logger
logger = logging.getLogger(__name__)

# Load model once at import time (can be moved to a separate init function if preferred)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
# Force float32 explicitly rather than relying on whatever dtype a given
# transformers version defaults to for this checkpoint. WhisperProcessor's
# feature extraction always produces float32 regardless, so if the model
# ends up in float16 while the input stays float32, every forward pass
# fails with a dtype mismatch (seen as "Input type (float) and bias type
# (Half) should be the same").
model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL, torch_dtype=torch.float32).to(device)
model.eval()
logger.info(f"Whisper model loaded: {WHISPER_MODEL} (dtype={model.dtype})")

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Convert audio bytes (WAV format) to French text using Whisper.
    """
    logger.info(f"Transcribing audio of {len(audio_bytes)} bytes")

    # Read audio from bytes using soundfile
    audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # Reject audio that exceeds the configured max duration. Checked here
    # (in addition to the client-side auto-stop) since the /transcribe
    # endpoint can be hit directly.
    duration_sec = len(audio_np) / float(sample_rate)
    if duration_sec > MAX_AUDIO_DURATION_SECONDS:
        raise ValueError(
            f"L'audio dépasse la durée maximale autorisée "
            f"({MAX_AUDIO_DURATION_SECONDS} secondes)."
        )

    waveform = torch.from_numpy(audio_np).float()

    # Ensure waveform is 2D: (channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)          # mono -> (1, samples)
    else:
        waveform = waveform.transpose(0, 1)       # (samples, channels) -> (channels, samples)

    # Resample to 16 kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Convert to mono by averaging channels if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert to numpy array (1D)
    audio_array = waveform.squeeze().numpy()

    # Process input and generate transcription. Cast to the model's actual
    # dtype (not just its device): feature extraction always produces
    # float32, so this only does something if the model isn't float32, but
    # reading it from the model itself rather than assuming keeps this
    # correct even if the loaded dtype ever changes later.
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
    inputs = inputs.to(device=device, dtype=model.dtype)
    with torch.no_grad():
        generated_ids = model.generate(inputs, language="french", task="transcribe")

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.info(f"Whisper transcription: {transcription}")

    return transcription