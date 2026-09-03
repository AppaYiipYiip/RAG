# llm_utils.py
import gc
import logging
from llama_cpp import Llama
from config import QWEN_MODEL_PATH

logger = logging.getLogger(__name__)

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        logger.info(f"Loading Qwen model from {QWEN_MODEL_PATH}")
        _llm = Llama(model_path=QWEN_MODEL_PATH, n_ctx=4096, n_threads=8, verbose=False)
    return _llm

def generate(prompt, max_tokens=512, temperature=0, stop=None):
    """Run LLM generation and return text."""
    llm = get_llm()
    output = llm(prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.95,
                 stop=stop, echo=False)
    return output["choices"][0]["text"]