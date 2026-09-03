# answer_generator.py
import logging
import llm_utils
import pandas as pd

logger = logging.getLogger(__name__)

def _dataframe_to_prompt(df):
    if df is None:
        return "Aucune donnée fournie."
    if df.empty:
        return "Résultat vide."
    # Convert to a clear markdown table (requires tabulate)
    try:
        table = df.to_markdown(index=False)
    except:
        table = df.to_string(index=False)
    return table

def generate_answer(question: str, df: pd.DataFrame, sql: str = "", previous_context: str = None) -> str:
    """Generate an answer based on the SQL result, using ONLY the provided data."""
    data_text = _dataframe_to_prompt(df)

    system_msg = (
        "Tu es un assistant qui répond en français à une question posée par un utilisateur, "
        "en te basant sur les données fournies ci-dessous et, si pertinent, sur l'historique de la "
        "conversation. Ne fais aucune supposition et n'invente aucune information qui ne soit pas "
        "présente dans les données ou dans l'historique. Si rien ne permet de répondre, dis-le clairement."
    )
    if previous_context:
        system_msg += f"\nHistorique de la conversation :\n{previous_context}"

    prompt = f"""<|im_start|>system
{system_msg}
<|im_end|>
<|im_start|>user
Question : {question}

Données (extrait) :
{data_text}
<|im_end|>
<|im_start|>assistant
"""
    answer = llm_utils.generate(prompt, max_tokens=300, temperature=0.1, stop=["<|im_end|>", "\n\n"]).strip()
    logger.info(f"Generated answer: {answer}")
    return answer

def generate_chat_response(question: str, history: list = None) -> str:
    """Generate a conversational response without using any database data."""
    history_text = ""
    if history:
        history_text = "\n".join([f"User: {h['question']}\nAssistant: {h['answer']}" for h in history[-3:]])
    system_msg = (
        "Tu es un assistant amical et utile. Réponds brièvement en français à la conversation."
    )
    if history_text:
        system_msg += f"\nHistorique récent :\n{history_text}"

    prompt = f"""<|im_start|>system
{system_msg}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
    response = llm_utils.generate(prompt, max_tokens=150, temperature=0.3, stop=["<|im_end|>", "\n\n"]).strip()
    logger.info(f"Chat response: {response}")
    return response