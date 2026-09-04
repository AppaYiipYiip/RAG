# nlp_to_sql.py
import logging
import re

import config
import llm_utils

logger = logging.getLogger(__name__)

# Dialect-specific syntax reminders injected into the SQL-generation prompt.
# This only changes what we TELL the model; database.py's actual connection
# (pyodbc) is still SQL-Server-specific, swapping to Postgres/MySQL/SQLite
# for real would also need a different driver/connection layer there. These
# cover the well-known basics for each engine, not every dialect's edge
# cases, treat them as a starting point to extend as you hit real gaps.
_DIALECT_RULES = {
    "mssql": """Rappels de syntaxe T-SQL (SQL Server) :
- SQL Server n'a PAS de clause LIMIT. Pour limiter le nombre de lignes, utilise "SELECT TOP N ... ORDER BY ...".
- Pour la pagination, utilise "OFFSET ... ROWS FETCH NEXT ... ROWS ONLY" (nécessite un ORDER BY).
- Les colonnes de type TEXT, NTEXT ou IMAGE ne peuvent PAS être utilisées directement dans GROUP BY, ORDER BY, DISTINCT, ou une comparaison : utilise CAST(colonne AS NVARCHAR(MAX)) si nécessaire.
- Utilise GETDATE() pour la date/heure actuelle, DATEPART(unité, colonne) pour extraire année/mois/jour.""",
    "postgres": """Rappels de syntaxe PostgreSQL :
- Utilise "LIMIT N" (et "OFFSET N" pour la pagination) pour limiter le nombre de lignes.
- Utilise NOW() ou CURRENT_TIMESTAMP pour la date/heure actuelle, EXTRACT(unité FROM colonne) pour extraire année/mois/jour.
- Les identifiants sensibles à la casse doivent être entourés de guillemets doubles ("colonne").""",
    "mysql": """Rappels de syntaxe MySQL :
- Utilise "LIMIT N" (et "LIMIT offset, N" pour la pagination) pour limiter le nombre de lignes.
- Utilise NOW() pour la date/heure actuelle, YEAR()/MONTH()/DAY() pour extraire une partie de date.
- Utilise des backticks (`colonne`) pour les identifiants réservés.""",
    "sqlite": """Rappels de syntaxe SQLite :
- Utilise "LIMIT N" (et "OFFSET N" pour la pagination) pour limiter le nombre de lignes.
- Utilise datetime('now') pour la date/heure actuelle, strftime('%Y'/'%m'/'%d', colonne) pour extraire une partie de date.
- SQLite est typé de façon dynamique ; les contraintes de type sont indicatives, pas strictement appliquées.""",
}

# Applies regardless of dialect: an unaliased self-join produces duplicate
# column names in the result (SELECT e1.nom, e2.nom FROM employees e1 JOIN
# employees e2 ...), which breaks anything downstream assuming unique
# columns. We also defend against this in database.py by renaming
# duplicates automatically, but avoiding it at the source is better.
_GENERAL_RULES = """Règle générale :
- Utilise toujours un alias (AS) pour chaque colonne sélectionnée plusieurs fois ou provenant d'une auto-jointure, afin d'éviter des noms de colonnes en double dans le résultat (ex: e1.nom AS nom_employe1, e2.nom AS nom_employe2)."""


def schema_to_prompt(schema: dict) -> str:
    """
    Render the schema for the prompt. Structural facts (table/column names,
    types, keys) always come from the live database via schema_loader.py,
    never hand-edited, so they can't go stale. If a matching entry exists
    in an optional hand-written enrichment file (schema_metadata.yaml, see
    schema_loader.load_schema_metadata), its "description"/"synonyms"/
    "example" fields are rendered alongside the structural facts. Missing
    enrichment for a table or column is fine, it just renders without the
    extra context, this is a pure addition, never a requirement.
    """
    lines = []
    for table in schema.get("tables", []):
        lines.append(f"Table: {table['name']}")
        if table.get("description"):
            lines.append(f"  ({table['description']})")
        lines.append("Columns:")
        for col in table.get("columns", []):
            pk = " PRIMARY KEY" if col.get("primary_key") else ""
            nullable = " NULL" if col.get("nullable") else " NOT NULL"
            identity = " IDENTITY" if col.get("identity") else ""
            extra_bits = []
            if col.get("description"):
                extra_bits.append(col["description"])
            if col.get("synonyms"):
                extra_bits.append("aussi appelé: " + ", ".join(col["synonyms"]))
            if col.get("example") is not None:
                extra_bits.append(f"exemple: {col['example']!r}")
            extra = f"  # {' | '.join(extra_bits)}" if extra_bits else ""
            lines.append(f"  - {col['name']} ({col['type']}){pk}{nullable}{identity}{extra}")
        fks = table.get("foreign_keys", [])
        if fks:
            lines.append("Foreign Keys:")
            for fk in fks:
                lines.append(f"  - {fk['column']} -> {fk['references_table']}.{fk['references_column']}")
    return "\n".join(lines)


def text_to_sql(question: str, schema: dict, history_text: str = "",
                 previous_sql: str = None, previous_error: str = None) -> str:
    schema_text = schema_to_prompt(schema)
    dialect_rules = _DIALECT_RULES.get(config.SQL_DIALECT, _DIALECT_RULES["mssql"])

    history_block = ""
    if history_text:
        history_block = f"""
Contexte de la conversation (une requête précédente peut être réutilisée ou adaptée si la question s'y rapporte) :
{history_text}
"""

    correction_block = ""
    if previous_sql and previous_error:
        correction_block = f"""
Une tentative précédente n'était pas satisfaisante, corrige-la :
Requête précédente : {previous_sql}
Problème identifié : {previous_error}
"""

    # No pre-filled ```sql here (unlike earlier): pre-filling the start of
    # the assistant's turn would block a thinking model from opening with
    # its <think> block, which has to come first. Instead we ask for the
    # fence explicitly and extract it afterward; harmless for a
    # non-thinking model too, it just writes the fence itself with no
    # preamble.
    prompt = f"""<|im_start|>system
Tu es un assistant qui convertit des questions en langage naturel en requêtes SQL ({config.SQL_DIALECT}).
Utilise uniquement le schéma fourni. Si la question fait suite à une requête précédente dans la conversation, appuie-toi dessus pour construire ou adapter la nouvelle requête.

{dialect_rules}

{_GENERAL_RULES}

Tu peux réfléchir avant de répondre. Termine toujours ta réponse par la requête SQL, et uniquement la requête SQL, entourée de balises ```sql et ```.
<|im_end|>
<|im_start|>user
Schéma de la base de données :
{schema_text}
{history_block}{correction_block}
Question : {question}
<|im_end|>
<|im_start|>assistant
"""
    raw_output = llm_utils.generate(
        prompt, max_tokens=config.REASONING_MAX_TOKENS, temperature=0,
        stop=["```", "<|im_end|>", "\n\n"], role="reasoning"
    )
    sql = raw_output.strip()
    if sql.startswith("```sql"):
        sql = sql[6:]
    elif sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    sql = sql.strip()
    if not sql:
        raise ValueError("Empty SQL generated.")
    return sql


def evaluate_sql_result(question: str, sql: str, df) -> dict:
    """
    Ask the model to sanity-check whether the query it just ran actually
    looks like it answers the question, catching cases that execute
    without error but are still wrong: an empty result from a bad filter,
    a missing join, the wrong column aggregated, and so on.

    Deliberately kept to a single verdict word rather than free-form
    reasoning: even a reasoning-capable model is more reliably parsed with
    a short discriminative call ("OK" vs "RETRY") than with open-ended
    self-analysis. Treat the result as a soft signal, not a guarantee.

    Returns {"verdict": "OK", "reason": ""} or {"verdict": "RETRY", "reason": "..."}.
    If the response can't be parsed, this fails open to OK rather than
    risking a loop that can never resolve.
    """
    row_count = 0 if df is None else len(df)
    if df is None or df.empty:
        preview = "(aucune ligne)"
    else:
        preview = df.head(5).to_string(index=False)

    prompt = f"""<|im_start|>system
Tu vérifies si le résultat d'une requête SQL répond correctement à la question posée.
Tu peux réfléchir avant de répondre, mais termine toujours par une seule ligne :
"OK" si le résultat semble correct et suffisant pour répondre à la question.
"RETRY: raison courte" si le résultat semble incorrect, vide de façon suspecte, ou ne répond pas vraiment à la question (mauvaise colonne, mauvais filtre, jointure manquante, etc.).
Ta toute dernière ligne doit être exactement "OK" ou "RETRY: ...", rien d'autre après.
<|im_end|>
<|im_start|>user
Question : {question}
Requête SQL exécutée : {sql}
Nombre de lignes retournées : {row_count}
Aperçu du résultat :
{preview}
<|im_end|>
<|im_start|>assistant
"""
    try:
        raw = llm_utils.generate(
            prompt, max_tokens=config.REASONING_SELFCHECK_MAX_TOKENS, temperature=0,
            stop=["<|im_end|>"], role="reasoning"
        ).strip()
    except Exception as e:
        logger.warning(f"SQL self-check call failed, defaulting to OK: {e}")
        return {"verdict": "OK", "reason": ""}

    # The verdict is asked for as the LAST line so a thinking model can
    # reason freely above it without breaking the parse. Word-boundary
    # search rather than a strict startswith, since a model that adds even
    # a small label ("Verdict : OK") would otherwise silently fail open
    # every time, masking a real RETRY as an accidental OK.
    last_line = raw.strip().splitlines()[-1].strip() if raw.strip() else ""
    upper = last_line.upper()
    if re.search(r"\bRETRY\b", upper):
        m = re.search(r"RETRY\b\s*:?\s*(.*)", last_line, re.IGNORECASE)
        reason = m.group(1).strip() if m else ""
        return {"verdict": "RETRY", "reason": reason or "Résultat jugé insatisfaisant par le modèle."}
    if re.search(r"\bOK\b", upper):
        return {"verdict": "OK", "reason": ""}

    logger.warning(f"Could not parse SQL self-check verdict, defaulting to OK: {raw!r}")
    return {"verdict": "OK", "reason": ""}
