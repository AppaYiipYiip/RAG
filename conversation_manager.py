# conversation_manager.py
import logging
import llm_utils
import pandas as pd

logger = logging.getLogger(__name__)

# How many recent turns are fed into each LLM call. Kept separate because
# the SQL generator prompt already carries the full schema, so it gets a
# tighter budget than the classifier or the answer generator.
MAX_HISTORY_TURNS = 8
MAX_CONTEXT_TURNS_CLASSIFY = 5
MAX_CONTEXT_TURNS_SQL = 3
MAX_CONTEXT_TURNS_ANSWER = 5


# How many turns (including their DataFrame, kept for on-demand chart
# requests) we retain in memory. Larger than the LLM context slices below,
# since a chart might be requested well after a turn scrolled out of prompt
# context but is still visible in the chat log.
MAX_STORED_TURNS = 20
MAX_CONTEXT_TURNS_CLASSIFY = 5
MAX_CONTEXT_TURNS_SQL = 3
MAX_CONTEXT_TURNS_ANSWER = 5


class ConversationState:
    def __init__(self):
        self.history = []           # list of turn dicts, oldest first
        self.next_turn_id = 1
        self.current_sql = None     # SQL of the most recently executed query
        self.current_df = None      # DataFrame of the most recently executed query
        self.current_question = None
        self.current_answer = None

state = ConversationState()


def clear_state():
    """Wipe the whole conversation: history (and every turn's DataFrame) and the active dataset."""
    state.history.clear()
    state.next_turn_id = 1
    state.current_sql = None
    state.current_df = None
    state.current_question = None
    state.current_answer = None


def add_turn(question, answer, sql=None, df=None, turn_type="chat"):
    """
    Append a turn to the conversation history and trim to MAX_STORED_TURNS.
    turn_type is "chat" for plain conversational turns or "sql" for turns
    backed by a query (whether newly generated or reused). The DataFrame is
    kept as-is (not just a summary) so a later on-demand chart request uses
    the exact data the answer was already built from. Returns the turn's id.
    """
    turn_id = state.next_turn_id
    state.next_turn_id += 1
    state.history.append({
        "id": turn_id,
        "type": turn_type,
        "question": question,
        "answer": answer,
        "sql": sql,
        "df": df,
        "df_summary": summarize_df(df) if df is not None else None,
    })
    if len(state.history) > MAX_STORED_TURNS:
        state.history = state.history[-MAX_STORED_TURNS:]
    return turn_id


def get_turn(turn_id):
    """Look up a stored turn by id, e.g. for an on-demand chart request. None if evicted or never existed."""
    for turn in state.history:
        if turn.get("id") == turn_id:
            return turn
    return None


def summarize_df(df, max_rows=5):
    if df is None:
        return "Aucun résultat"
    if df.empty:
        return "Résultat vide"
    cols = ", ".join(df.columns)
    rows = df.head(max_rows).to_string(index=False)
    return f"Colonnes: {cols}\n{rows}"


def format_history(turns) -> str:
    """Render a list of history turns as a readable transcript for prompts."""
    lines = []
    for t in turns:
        lines.append(f"Utilisateur : {t['question']}")
        if t["type"] == "sql" and t.get("sql"):
            lines.append(f"SQL utilisée : {t['sql']}")
            if t.get("df_summary"):
                lines.append(f"Résultat : {t['df_summary']}")
        lines.append(f"Assistant : {t['answer']}")
    return "\n".join(lines)


def get_last_exchange():
    if state.history:
        return state.history[-1]
    return None


def decide_action(question: str) -> str:
    """
    Classify the user's message into:
    - CHAT: greeting, thanks, or a question that needs no data.
    - NEW_QUERY: needs a fresh SQL query against the database.
    - FOLLOW_UP: can be answered from the most recently executed query's
      result, without running new SQL.
    """
    recent = state.history[-MAX_CONTEXT_TURNS_CLASSIFY:]
    context = f"\nHistorique récent :\n{format_history(recent)}\n" if recent else ""

    active_query_note = ""
    if state.current_sql:
        active_query_note = (
            f"\nRequête SQL actuellement active (dernier résultat disponible) :\n{state.current_sql}\n"
        )

    prompt = f"""<|im_start|>system
Tu es un assistant qui classifie les messages d'un utilisateur en trois catégories :
- CHAT : le message est une salutation, un remerciement, ou une question qui ne nécessite pas de données (ex: "bonjour", "merci", "comment ça va ?").
- NEW_QUERY : le message demande une information qui n'est pas déjà disponible dans le dernier résultat et nécessite une nouvelle requête SQL.
- FOLLOW_UP : le message peut être répondu en utilisant uniquement le dernier résultat SQL déjà obtenu (ex: reformulation, calcul simple sur ce résultat, précision).

Réponds uniquement par le nom de la catégorie (CHAT, NEW_QUERY, ou FOLLOW_UP).
<|im_end|>
<|im_start|>user
{context}{active_query_note}
Nouveau message de l'utilisateur : "{question}"
<|im_end|>
<|im_start|>assistant
"""
    response = llm_utils.generate(prompt, max_tokens=10, temperature=0, stop=["<|im_end|>", "\n"], role="chat").strip().upper()
    logger.info(f"Decision for '{question}': {response}")
    if "FOLLOW_UP" in response:
        return "FOLLOW_UP"
    elif "CHAT" in response:
        return "CHAT"
    else:
        return "NEW_QUERY"
