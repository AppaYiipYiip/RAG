import os
from dotenv import load_dotenv

load_dotenv()

DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_NAME = os.getenv("DB_NAME", "MyInternProject")
DB_USER = os.getenv("DB_USER", "intern_dev")
DB_PASSWORD = os.getenv("DB_PASSWORD", "YourStrongP@ssw0rd!")

# Which SQL dialect the connected database speaks. Drives which syntax
# reminders get injected into the SQL-generation prompt in nlp_to_sql.py
# (TOP vs LIMIT, date functions, quoting, etc). This only changes what we
# tell the model, not which Python driver database.py uses to connect,
# database.py is still pyodbc/SQL-Server-specific; swapping to Postgres or
# MySQL would need a different connection layer too, not just this setting.
SQL_DIALECT = os.getenv("SQL_DIALECT", "mssql")  # mssql | postgres | mysql | sqlite

# Optional path to a hand-written schema enrichment file (table/column
# descriptions, synonyms, example values) layered on top of the
# auto-introspected schema. See schema_metadata.example.yaml for the
# format. If the file doesn't exist, the app runs fine without it, this is
# a pure enrichment, never a requirement.
SCHEMA_METADATA_PATH = os.getenv("SCHEMA_METADATA_PATH", "schema_metadata.yaml")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "bofenghuang/whisper-large-v3-french")

# Hard cap on recorded audio length accepted by /transcribe, in seconds.
# Keep this in sync with the client-side auto-stop timer in static/script.js.
MAX_AUDIO_DURATION_SECONDS = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "30"))

# How many times we let the LLM try to generate working SQL for one question
# before giving up. On failure (syntax error, rejected as non-read-only, or
# the model's own semantic self-check flagging the result), the error or
# critique is fed back to the model so it can self-correct. Raised from 3 to
# 5: retrying costs latency but nothing else, so there's little reason to be
# stingy with it. Note this alone won't fix a failure the model doesn't
# understand (see the TEXT/NTEXT GROUP BY case), more attempts only help
# when the model has a real chance of doing something different next time.
MAX_SQL_GENERATION_ATTEMPTS = int(os.getenv("MAX_SQL_GENERATION_ATTEMPTS", "5"))

# Same idea for chart spec generation, also raised to 5.
MAX_CHART_GENERATION_ATTEMPTS = int(os.getenv("MAX_CHART_GENERATION_ATTEMPTS", "5"))

# Verbosity of Python's logging module across the whole app (root logger),
# so this also affects Flask/Werkzeug/transformers' own logging, not just
# our code. INFO is the normal running level; DEBUG additionally logs every
# full prompt sent to the LLM and its full raw response (see llm_utils.py),
# which is verbose but is exactly what you want when trying to see why a
# specific answer came out wrong.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# --- LLM role configuration --------------------------------------------
# Every LLM call in the app goes through llm_utils.generate(..., role=...).
# Two roles are used:
#   "reasoning" - SQL generation, the SQL self-check, chart spec generation,
#                 and the chart self-check. Structured, verifiable-ish
#                 output; a good match for a reasoning-tuned model.
#   "chat"      - conversation classification, chat replies, and the final
#                 French answer text. Open-ended conversational output.
#
# Each role independently supports:
#   backend    "local" (a GGUF file loaded in-process via llama-cpp-python)
#              or "api" (any OpenAI-compatible /chat/completions endpoint,
#              hosted or self-hosted, e.g. llama-server, vLLM, Ollama).
#   is_thinking  whether this model wraps reasoning in <think>...</think>
#              before its answer (see llm_utils._strip_thinking). Thinking
#              models need a much larger max_tokens budget than a direct-
#              answer model, they can spend hundreds to thousands of tokens
#              reasoning before writing anything else; too small a budget
#              means generation gets cut off mid-thought and no answer is
#              ever recovered.
#
# To collapse back to a single model for everything, point both roles at
# the same model_path (or the same api_base/api_model) and set is_thinking
# to match that one model.

REASONING_MODEL_PATH = os.getenv("REASONING_MODEL_PATH", "VibeThinker-3B-Q4_K_M.gguf")
CHAT_MODEL_PATH = os.getenv("CHAT_MODEL_PATH", "qwen2.5-coder-3b-instruct-q4_k_m.gguf")

# Context window and token budgets. 8192/2048 here are a deliberately
# moderate starting point, not the model's real ceiling (VibeThinker-3B
# supports far more, but a much larger n_ctx costs real RAM for the KV
# cache). Raise these if you see truncated-thinking warnings in the logs
# and have RAM to spare; our SQL/chart questions are far simpler than the
# competition-math problems this model was benchmarked on, so they likely
# don't need anywhere near its full budget.
REASONING_N_CTX = int(os.getenv("REASONING_N_CTX", "8192"))
REASONING_MAX_TOKENS = int(os.getenv("REASONING_MAX_TOKENS", "4096"))
REASONING_SELFCHECK_MAX_TOKENS = int(os.getenv("REASONING_SELFCHECK_MAX_TOKENS", "1024"))

CHAT_N_CTX = int(os.getenv("CHAT_N_CTX", "4096"))

LLM_ROLES = {
    "reasoning": {
        "backend": os.getenv("REASONING_BACKEND", "local"),
        "model_path": REASONING_MODEL_PATH,
        "n_ctx": REASONING_N_CTX,
        "n_threads": int(os.getenv("REASONING_N_THREADS", "8")),
        "n_gpu_layers": int(os.getenv("REASONING_N_GPU_LAYERS", "0")),
        "is_thinking": os.getenv("REASONING_IS_THINKING", "true").lower() == "true",
        "thinking_tag": os.getenv("REASONING_THINKING_TAG", "think"),
        "api_base": os.getenv("REASONING_API_BASE", ""),
        "api_key": os.getenv("REASONING_API_KEY", ""),
        "api_model": os.getenv("REASONING_API_MODEL", ""),
    },
    "chat": {
        "backend": os.getenv("CHAT_BACKEND", "local"),
        "model_path": CHAT_MODEL_PATH,
        "n_ctx": CHAT_N_CTX,
        "n_threads": int(os.getenv("CHAT_N_THREADS", "8")),
        "n_gpu_layers": int(os.getenv("CHAT_N_GPU_LAYERS", "0")),
        "is_thinking": os.getenv("CHAT_IS_THINKING", "false").lower() == "true",
        "thinking_tag": os.getenv("CHAT_THINKING_TAG", "think"),
        "api_base": os.getenv("CHAT_API_BASE", ""),
        "api_key": os.getenv("CHAT_API_KEY", ""),
        "api_model": os.getenv("CHAT_API_MODEL", ""),
    },
}
