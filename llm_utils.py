# llm_utils.py
#
# Single entry point for every LLM call in the app. Every other module
# calls generate(prompt, ..., role="reasoning" | "chat") and never talks to
# llama_cpp or an HTTP API directly. This is the ONLY file that needs to
# change to swap which model handles which kind of work, move a role from
# local to a remote API, or add a new role, everything else in the app is
# unaffected by that choice.
#
# Two roles are used by the rest of the app:
#   "reasoning" - SQL generation, the SQL self-check, chart spec generation,
#                 and the chart self-check. Structured, verifiable-ish
#                 output, a good match for a reasoning-tuned model.
#   "chat"      - conversation classification, chat replies, and the final
#                 French answer text. Open-ended conversational output.
# Point both roles at the same model_path in config.py to collapse back to
# a single model handling everything.

import logging
import time

import config

logger = logging.getLogger(__name__)

_local_models = {}  # model_path -> loaded llama_cpp.Llama instance, lazy per path


def _get_local_model(role_config: dict):
    """Lazily load and cache a local GGUF model. Reused across roles if two
    roles happen to point at the same model_path."""
    path = role_config["model_path"]
    if path not in _local_models:
        from llama_cpp import Llama
        n_ctx = role_config.get("n_ctx", 4096)
        logger.info(f"Loading local model: {path} (n_ctx={n_ctx})")
        start = time.monotonic()
        _local_models[path] = Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_threads=role_config.get("n_threads", 8),
            n_gpu_layers=role_config.get("n_gpu_layers", 0),
            verbose=False,
        )
        logger.info(f"Model loaded in {time.monotonic() - start:.1f}s: {path}")
    return _local_models[path]


def _strip_thinking(text: str, tag: str = "think") -> str:
    """
    Remove a leading <tag>...</tag> reasoning block (used by VibeThinker,
    DeepSeek-R1-style models, and similar). Reasoning always comes first in
    these models, so we only ever look for a block at the very start.

    If the closing tag is missing, generation was almost certainly cut off
    mid-thought by too small a max_tokens budget. In that case we return an
    empty string rather than a stray unclosed fragment: an empty result is
    an obvious, loud signal (every downstream parser will visibly fail to
    find what it expected) that the budget needs raising, whereas silently
    returning half a reasoning trace could masquerade as a bad answer and
    be much more confusing to debug.
    """
    open_tag, close_tag = f"<{tag}>", f"</{tag}>"
    stripped = text.strip()
    if not stripped.startswith(open_tag):
        return text
    end = stripped.find(close_tag)
    if end == -1:
        logger.warning(
            f"<{tag}> block never closed, likely truncated by max_tokens. "
            f"Dropping it all rather than guessing; raise the token budget for this role. "
            f"({len(stripped)} raw chars)"
        )
        return ""
    return stripped[end + len(close_tag):].strip()


def _call_local(role_config: dict, prompt: str, max_tokens: int, temperature: float, stop) -> str:
    llm = _get_local_model(role_config)
    # Defensive measure: force a clean internal state before every call.
    # Our prompts are always independent, single-turn requests (we build the
    # full prompt text ourselves each time, never relying on llama.cpp's own
    # multi-turn state), so there's never legitimate prefix-cache reuse to
    # lose here. This guards against cross-call KV-cache state drifting
    # across a long chain of unrelated prompt shapes in one process, which
    # is a known category of issue in llama-cpp-python and matches a native
    # crash (GGML_ASSERT on an out-of-bounds index) observed in testing.
    if hasattr(llm, "reset"):
        llm.reset()
    output = llm(prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.95,
                 stop=stop, echo=False)
    return output["choices"][0]["text"]


def _call_api(role_config: dict, prompt: str, max_tokens: int, temperature: float, stop) -> str:
    """POSTs an OpenAI-compatible /chat/completions request. Works against
    any server implementing that shape: a hosted API, or a local server
    like llama.cpp's own `llama-server`, vLLM, or Ollama's OpenAI-compatible
    endpoint, so "local" vs "api" is really "in-process" vs "over HTTP",
    not "can't be your own machine"."""
    import requests
    url = role_config["api_base"].rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if role_config.get("api_key"):
        headers["Authorization"] = f"Bearer {role_config['api_key']}"
    payload = {
        "model": role_config.get("api_model", ""),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        payload["stop"] = stop
    response = requests.post(url, headers=headers, json=payload, timeout=role_config.get("timeout", 120))
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate(prompt: str, max_tokens: int = 512, temperature: float = 0, stop=None, role: str = "chat") -> str:
    """
    Run LLM generation for the given role. Returns the final text with any
    <think>...</think> block already stripped if that role is configured as
    a thinking model (see config.LLM_ROLES). Logs timing at INFO and the
    full raw prompt/output at DEBUG.
    """
    role_config = config.LLM_ROLES.get(role, config.LLM_ROLES["chat"])
    backend = role_config.get("backend", "local")
    is_thinking = role_config.get("is_thinking", False)

    # Content-specific stop sequences (blank line, a stray ```) are meant to
    # end a direct answer, but can appear naturally INSIDE a reasoning
    # trace and would truncate the model before it ever reaches its actual
    # answer. Thinking roles only stop on the real turn-end marker; fence
    # stripping and thinking stripping both happen as post-processing on
    # the complete text instead.
    effective_stop = ["<|im_end|>"] if is_thinking else stop

    logger.debug(f"[{role}] prompt ({len(prompt)} chars):\n{prompt}")
    start = time.monotonic()

    if backend == "local":
        raw_text = _call_local(role_config, prompt, max_tokens, temperature, effective_stop)
    elif backend == "api":
        raw_text = _call_api(role_config, prompt, max_tokens, temperature, effective_stop)
    else:
        raise ValueError(f"Unknown backend for role {role!r}: {backend!r}")

    elapsed = time.monotonic() - start
    logger.info(f"[{role}] generation finished in {elapsed:.2f}s ({len(raw_text)} raw chars, backend={backend})")
    logger.debug(f"[{role}] raw output:\n{raw_text!r}")

    if is_thinking:
        text = _strip_thinking(raw_text, tag=role_config.get("thinking_tag", "think"))
        logger.debug(f"[{role}] after stripping <{role_config.get('thinking_tag', 'think')}>: {len(text)} chars")
        return text
    return raw_text


def warm_up(role: str):
    """Eagerly load a role's local model so the first real request isn't
    slow. No-op for an API-backed role, there's nothing local to load."""
    role_config = config.LLM_ROLES.get(role, config.LLM_ROLES["chat"])
    if role_config.get("backend", "local") == "local":
        _get_local_model(role_config)
