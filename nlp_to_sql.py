# nlp_to_sql.py
import logging
import llm_utils

logger = logging.getLogger(__name__)

def schema_to_prompt(schema: dict) -> str:
    # same as before
    lines = []
    for table in schema.get("tables", []):
        lines.append(f"Table: {table['name']}")
        lines.append("Columns:")
        for col in table.get("columns", []):
            pk = " PRIMARY KEY" if col.get("primary_key") else ""
            nullable = " NULL" if col.get("nullable") else " NOT NULL"
            identity = " IDENTITY" if col.get("identity") else ""
            lines.append(f"  - {col['name']} ({col['type']}){pk}{nullable}{identity}")
        fks = table.get("foreign_keys", [])
        if fks:
            lines.append("Foreign Keys:")
            for fk in fks:
                lines.append(f"  - {fk['column']} -> {fk['references_table']}.{fk['references_column']}")
    return "\n".join(lines)

def text_to_sql(question: str, schema: dict, history_text: str = "") -> str:
    schema_text = schema_to_prompt(schema)
    history_block = ""
    if history_text:
        history_block = f"""
Contexte de la conversation (une requête précédente peut être réutilisée ou adaptée si la question s'y rapporte) :
{history_text}
"""
    prompt = f"""<|im_start|>system
Tu es un assistant qui convertit des questions en langage naturel en requêtes SQL Server.
Utilise uniquement le schéma fourni. Si la question fait suite à une requête précédente dans la conversation, appuie-toi dessus pour construire ou adapter la nouvelle requête.
Produis seulement la requête SQL, sans explication, sans commentaire.
<|im_end|>
<|im_start|>user
Schéma de la base de données :
{schema_text}
{history_block}
Question : {question}
<|im_end|>
<|im_start|>assistant
```sql
"""
    raw_output = llm_utils.generate(prompt, max_tokens=512, temperature=0, stop=["```", "<|im_end|>", "\n\n"])
    sql = raw_output.strip()
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.endswith("```"):
        sql = sql[:-3]
    sql = sql.strip()
    if not sql:
        raise ValueError("Empty SQL generated.")
    return sql