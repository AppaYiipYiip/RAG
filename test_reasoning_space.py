"""
test_reasoning_space.py

Direct functional test of the reasoning Space: not just "does it
respond", but "can it actually do the two jobs this role has in
production" - generating SQL against a schema, and choosing a chart
spec for a result. Uses the same dialect rules and chart-spec
instructions as nlp_to_sql.py and graph_generator.py, word for word,
so a pass here is a real signal about the deployed model, not a toy
"hello world" check.

Fill in SPACE_URL and API_KEY below, then:
    python test_reasoning_space.py
"""

import json
import re
import time
import requests

SPACE_URL = "https://AppaYiipYiip-reasoning-server.hf.space"
API_KEY = "TestKey1234567890"

ENDPOINT = SPACE_URL.rstrip("/") + "/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# ---------------------------------------------------------------------
# Copied verbatim from nlp_to_sql.py, so this test is grounded in the
# real production prompt, not a simplified stand-in.
# ---------------------------------------------------------------------
DIALECT_RULES = """Rappels de syntaxe T-SQL (SQL Server) :
- SQL Server n'a PAS de clause LIMIT. Pour limiter le nombre de lignes, utilise "SELECT TOP N ... ORDER BY ...".
- Pour la pagination, utilise "OFFSET ... ROWS FETCH NEXT ... ROWS ONLY" (nécessite un ORDER BY).
- Les colonnes de type TEXT, NTEXT ou IMAGE ne peuvent PAS être utilisées directement dans GROUP BY, ORDER BY, DISTINCT, ou une comparaison : utilise CAST(colonne AS NVARCHAR(MAX)) si nécessaire.
- Utilise GETDATE() pour la date/heure actuelle, DATEPART(unité, colonne) pour extraire année/mois/jour."""

GENERAL_RULES = """Règle générale :
- Utilise toujours un alias (AS) pour chaque colonne sélectionnée plusieurs fois ou provenant d'une auto-jointure, afin d'éviter des noms de colonnes en double dans le résultat (ex: e1.nom AS nom_employe1, e2.nom AS nom_employe2)."""

FAKE_SCHEMA = """Table: employees
Columns:
  - employees_id (int) PRIMARY KEY NOT NULL
  - nom (nvarchar) NOT NULL
  - poste (nvarchar) NOT NULL  # Intitulé du poste occupé par l'employé.
  - salaire (float) NOT NULL  # Salaire annuel brut, en euros.
Table: commandes
Columns:
  - commande_id (int) PRIMARY KEY NOT NULL
  - employees_id (int) NOT NULL
  - date_commande (date) NOT NULL
  - montant_total (float) NOT NULL  # Montant total de la commande, en euros.
Foreign Keys:
  - employees_id -> employees.employees_id"""

SQL_QUESTION = "Quel est le montant total des commandes pour chaque employé, du plus élevé au moins élevé ?"

# ---------------------------------------------------------------------
# Copied verbatim from graph_generator.CHART_SPEC_DOC, examples
# included, since the model pattern-matching against these same
# examples is exactly how it behaves in production too.
# ---------------------------------------------------------------------
CHART_SPEC_DOC = """Tu dois répondre avec un objet JSON décrivant comment tracer un graphique à partir des données fournies. Format attendu :

{
  "chart_type": "bar" | "line" | "scatter" | "pie" | "histogram",
  "x": "nom_de_colonne",
  "y": "nom_de_colonne" ou ["colonne1", "colonne2"] pour plusieurs séries (null pour histogram),
  "series": "nom_de_colonne" ou null,
  "barmode": "group" | "stack" (uniquement pertinent si "series" est utilisé avec "bar"),
  "group_by": "nom_de_colonne" ou null,
  "aggregation": "sum" | "avg" | "count" | "min" | "max" | "none",
  "title": "titre court du graphique",
  "x_label": "étiquette de l'axe X",
  "y_label": "étiquette de l'axe Y"
}

Règles :
- Utilise UNIQUEMENT des noms de colonnes qui existent réellement dans les données fournies ci-dessous.
- "histogram" n'a pas besoin de "y" (laisse-le à null).
- "pie" ne devrait être utilisé que si la colonne "x" a peu de catégories (6 ou moins) ; sinon préfère "bar".
- Si les données sont déjà agrégées (une ligne par catégorie), utilise "aggregation": "none".
- N'utilise "group_by" avec une "aggregation" que s'il faut vraiment combiner plusieurs lignes par catégorie.
- Pour un graphique en barres avec une catégorie principale ET une sous-catégorie (ex: une barre groupée/"clustered" par client montrant chaque produit séparément à côté), utilise "x" pour la catégorie principale, "series" pour la sous-catégorie, et "barmode": "group". Utilise "barmode": "stack" seulement si l'utilisateur veut voir la somme empilée plutôt que la comparaison côte à côte.

Exemple :
Question : "Répartition des ventes par région"
Colonnes : region (object, 4 valeurs: 'Nord', 'Sud', 'Est', 'Ouest'), total_ventes (float64)
Réponse :
{"chart_type": "bar", "x": "region", "y": "total_ventes", "series": null, "barmode": null, "group_by": null, "aggregation": "none", "title": "Ventes par région", "x_label": "Région", "y_label": "Ventes totales (€)"}

Tu peux réfléchir avant de répondre. Termine toujours ta réponse par uniquement l'objet JSON, sans texte après."""

CHART_QUESTION = "Montant total des commandes par employé"
CHART_COLUMNS = "  - nom (object, 5 valeurs: 'Dupont', 'Martin', 'Bernard', 'Petit', 'Durand')\n  - montant_total (float64)"
CHART_PREVIEW = """nom     montant_total
Dupont          15420.50
Martin          12100.00
Bernard          9800.75
Petit            8500.00
Durand           7200.25"""


def call_reasoning(prompt: str, max_tokens: int = 4096, timeout: int = 180) -> str:
    """
    Same call shape as llm_utils._call_api for a thinking role: only stop
    on the real turn-end marker, a content-specific stop sequence could
    truncate VibeThinker mid-thought, same reasoning as llm_utils.generate().
    """
    payload = {
        "model": "reasoning",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stop": ["<|im_end|>"],
    }
    print("  Sending request, VibeThinker can take a while to finish thinking...")
    start = time.monotonic()
    response = requests.post(ENDPOINT, headers=HEADERS, json=payload, timeout=timeout)
    response.raise_for_status()
    elapsed = time.monotonic() - start
    text = response.json()["choices"][0]["message"]["content"]
    print(f"  Got a response in {elapsed:.1f}s ({len(text)} raw chars)")
    return text


def strip_thinking(text: str) -> str:
    """Same logic as llm_utils._strip_thinking: only the content after a
    leading <think>...</think> block counts as the actual answer."""
    stripped = text.strip()
    if not stripped.startswith("<think>"):
        return stripped
    end = stripped.find("</think>")
    if end == -1:
        print("  WARNING: <think> block never closed, likely truncated by max_tokens.")
        return ""
    return stripped[end + len("</think>"):].strip()


def extract_sql(raw_answer: str) -> str:
    sql = raw_answer.strip()
    if sql.startswith("```sql"):
        sql = sql[6:]
    elif sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]
    return sql.strip()


def report(checks: dict) -> bool:
    all_passed = True
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        all_passed = all_passed and passed
    return all_passed


def test_sql() -> bool:
    print("\n=== Test 1: SQL generation ===")
    prompt = f"""<|im_start|>system
Tu es un assistant qui convertit des questions en langage naturel en requêtes SQL (mssql).
Utilise uniquement le schéma fourni.

{DIALECT_RULES}

{GENERAL_RULES}

Tu peux réfléchir avant de répondre. Termine toujours ta réponse par la requête SQL, et uniquement la requête SQL, entourée de balises ```sql et ```.
<|im_end|>
<|im_start|>user
Schéma de la base de données :
{FAKE_SCHEMA}

Question : {SQL_QUESTION}
<|im_end|>
<|im_start|>assistant
"""
    raw = call_reasoning(prompt)
    answer = strip_thinking(raw)
    sql = extract_sql(answer)

    print("\n  --- Generated SQL ---")
    print(" ", sql.replace("\n", "\n  "))
    print("  ----------------------")

    checks = {
        "Contains SELECT": "select" in sql.lower(),
        "Contains GROUP BY (question needs one row per employee)": "group by" in sql.lower(),
        "Does NOT use LIMIT (that's Postgres/MySQL syntax, not SQL Server)": "limit" not in sql.lower(),
        "Uses TOP or ORDER BY (correct MSSQL way to rank/limit results)": "top" in sql.lower() or "order by" in sql.lower(),
    }
    return report(checks)


def test_chart_spec() -> bool:
    print("\n=== Test 2: Chart spec generation ===")
    prompt = f"""<|im_start|>system
Tu es un assistant qui choisit comment représenter graphiquement le résultat d'une requête SQL.
{CHART_SPEC_DOC}
<|im_end|>
<|im_start|>user
Question de l'utilisateur : {CHART_QUESTION}

Colonnes disponibles dans les données :
{CHART_COLUMNS}

Aperçu des données (5 premières lignes) :
{CHART_PREVIEW}
<|im_end|>
<|im_start|>assistant
"""
    raw = call_reasoning(prompt)
    answer = strip_thinking(raw)

    print("\n  --- Raw response tail ---")
    print(" ", answer[-500:].replace("\n", "\n  "))
    print("  --------------------------")

    match = re.search(r"\{.*\}", answer, re.DOTALL)
    spec = None
    if match:
        try:
            spec = json.loads(match.group(0))
        except json.JSONDecodeError:
            spec = None

    checks = {"Response contains a parseable JSON object": spec is not None}
    if spec:
        checks["chart_type is one of the allowed values"] = spec.get("chart_type") in {
            "bar", "line", "scatter", "pie", "histogram"
        }
        checks["x matches a real column (nom)"] = spec.get("x") == "nom"
        checks["y matches a real column (montant_total)"] = spec.get("y") == "montant_total"
        print(f"\n  Parsed spec: {json.dumps(spec, ensure_ascii=False)}")

    return report(checks)


if __name__ == "__main__":
    if SPACE_URL == "https://REPLACE-ME.hf.space" or API_KEY == "REPLACE-ME":
        print("Fill in SPACE_URL and API_KEY at the top of this file first.")
        raise SystemExit(1)

    sql_ok = test_sql()
    chart_ok = test_chart_spec()

    print("\n=== Summary ===")
    print(f"SQL generation:   {'PASS' if sql_ok else 'CHECK OUTPUT ABOVE'}")
    print(f"Chart generation: {'PASS' if chart_ok else 'CHECK OUTPUT ABOVE'}")
    print("\nAutomated checks are a sanity net, not a quality judgment, read the")
    print("actual generated SQL and chart spec above yourself before trusting them.")
