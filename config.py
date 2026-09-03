import os
from dotenv import load_dotenv

load_dotenv()

DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_NAME = os.getenv("DB_NAME", "MyInternProject")
DB_USER = os.getenv("DB_USER", "intern_dev")
DB_PASSWORD = os.getenv("DB_PASSWORD", "YourStrongP@ssw0rd!")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "bofenghuang/whisper-large-v3-french")
QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "qwen2.5-coder-3b-instruct-q4_k_m.gguf")

# Hard cap on recorded audio length accepted by /transcribe, in seconds.
# Keep this in sync with the client-side auto-stop timer in static/script.js.
MAX_AUDIO_DURATION_SECONDS = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "30"))