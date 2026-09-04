# graph_generator.py
#
# Design note: the LLM never touches actual data values. It only picks a
# chart type and which columns map to which axes, as a small JSON object.
# That spec is validated against the real DataFrame's columns, then a plain
# pandas/plotly function builds the figure straight from the DataFrame.
# Every plotted number therefore comes from the SQL result, never from the
# model, the same "propose intent, validate, execute deterministically"
# pattern already used for text_to_sql / execute_sql.

import json
import logging
import re

import pandas as pd
import plotly.express as px

import config
import llm_utils

logger = logging.getLogger(__name__)

ALLOWED_CHART_TYPES = {"bar", "line", "scatter", "pie", "histogram"}
ALLOWED_AGGREGATIONS = {"sum", "avg", "count", "min", "max", "none"}
ALLOWED_BARMODES = {"group", "stack"}

# Above this many distinct categories a pie chart stops being readable;
# we fall back to a bar chart automatically rather than trusting the model
# to always follow the instruction below.
MAX_PIE_CATEGORIES = 8

# For a non-numeric column with few enough distinct values, list them
# outright in the prompt rather than just the dtype, concrete category
# names ground the model far better than "object" ever could, and this
# also tells it up front how many categories exist, the same number our
# own MAX_PIE_CATEGORIES safety net checks, so it can make a good pie-vs-bar
# call itself instead of relying on our fallback to fix it after the fact.
MAX_LISTED_UNIQUE_VALUES = 15

_AGG_FUNCS = {"sum": "sum", "avg": "mean", "count": "count", "min": "min", "max": "max"}

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

Exemple avec sous-catégorie (barres groupées) :
Question : "Quels produits achètent mes 5 meilleurs clients ?"
Colonnes : client (object, 5 valeurs: 'Client A', 'Client B', 'Client C', 'Client D', 'Client E'), produit (object, 6 valeurs: 'Produit X', 'Produit Y', ...), montant (float64)
Réponse :
{"chart_type": "bar", "x": "client", "y": "montant", "series": "produit", "barmode": "group", "group_by": null, "aggregation": "none", "title": "Achats par client et par produit", "x_label": "Client", "y_label": "Montant dépensé (€)"}

Tu peux réfléchir avant de répondre. Termine toujours ta réponse par uniquement l'objet JSON, sans texte après."""


def _build_column_lookup(schema: dict) -> dict:
    """
    Flatten the schema into column_name -> column dict, for a best-effort
    lookup against a query result's column names. Best-effort because an
    aliased result column (SUM(...) AS total_vente) won't match the
    original schema column name, this only helps when the LLM kept the
    original name or an alias that happens to coincide with it.
    """
    lookup = {}
    for table in schema.get("tables", []):
        for col in table.get("columns", []):
            if col.get("description") or col.get("synonyms"):
                lookup[col["name"]] = col
    return lookup


def _describe_columns(df: pd.DataFrame, schema: dict = None) -> str:
    """
    One line per column: name, dtype, and either the actual distinct
    values (if few enough to be useful context) or just a count. Built
    from df.dtypes.items() rather than looping df[col].dtype: selecting a
    duplicate-named column with df[col] returns a DataFrame instead of a
    Series, which has no .dtype attribute, this crashed in testing on a
    self-join result before database.py's column deduplication was added,
    df.dtypes is immune to that regardless, since it reads all columns at
    once rather than selecting one by (possibly ambiguous) label.

    If a schema (with schema_metadata.yaml enrichment merged in) is
    passed, a matching column's human description/synonyms are appended.
    """
    lookup = _build_column_lookup(schema) if schema else {}
    lines = []
    for col, dtype in df.dtypes.items():
        meta = lookup.get(col)
        hint = ""
        if meta:
            parts = []
            if meta.get("description"):
                parts.append(meta["description"])
            if meta.get("synonyms"):
                parts.append("aussi appelé: " + ", ".join(meta["synonyms"]))
            if parts:
                hint = f" [{'; '.join(parts)}]"

        is_numeric = pd.api.types.is_numeric_dtype(dtype)
        if not is_numeric:
            uniques = df[col].dropna().unique()
            if 0 < len(uniques) <= MAX_LISTED_UNIQUE_VALUES:
                values_text = ", ".join(repr(v) for v in uniques[:MAX_LISTED_UNIQUE_VALUES])
                lines.append(f"  - {col} ({dtype}, {len(uniques)} valeurs: {values_text}){hint}")
                continue
            elif len(uniques) > MAX_LISTED_UNIQUE_VALUES:
                lines.append(f"  - {col} ({dtype}, {len(uniques)} valeurs distinctes){hint}")
                continue
        lines.append(f"  - {col} ({dtype}){hint}")
    return "\n".join(lines)


def _build_prompt(question: str, answer: str, sql: str, df: pd.DataFrame,
                   previous_spec: dict = None, previous_feedback: str = None,
                   schema: dict = None) -> str:
    columns_block = _describe_columns(df, schema)
    preview = df.head(5).to_string(index=False)

    correction_block = ""
    if previous_spec and previous_feedback:
        correction_block = f"""
Une tentative précédente n'était pas satisfaisante, essaie un choix différent :
Choix précédent : {json.dumps(previous_spec, ensure_ascii=False)}
Problème identifié : {previous_feedback}
"""

    return f"""<|im_start|>system
Tu es un assistant qui choisit comment représenter graphiquement le résultat d'une requête SQL, pour aider un utilisateur à mieux comprendre la réponse à sa question.
{CHART_SPEC_DOC}
<|im_end|>
<|im_start|>user
Question de l'utilisateur : {question}

Réponse déjà donnée à l'utilisateur : {answer}

Requête SQL qui a produit ces données (utile pour comprendre ce que représente chaque colonne, par exemple après une auto-jointure) :
{sql}

Nombre total de lignes : {len(df)}

Colonnes disponibles dans les données :
{columns_block}

Aperçu des données (5 premières lignes) :
{preview}
{correction_block}
<|im_end|>
<|im_start|>assistant
"""


def _extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    # Tolerate stray text around the object (e.g. a leading "Voici :").
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def _validate_spec(spec: dict, df: pd.DataFrame) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("La réponse du modèle n'est pas un objet JSON valide.")

    chart_type = spec.get("chart_type")
    if chart_type not in ALLOWED_CHART_TYPES:
        raise ValueError(f"Type de graphique non supporté : {chart_type!r}")

    columns = set(df.columns)

    x = spec.get("x")
    if x is not None and x not in columns:
        raise ValueError(f"Colonne inconnue pour l'axe X : {x!r}")

    y = spec.get("y")
    if isinstance(y, list):
        for col in y:
            if col not in columns:
                raise ValueError(f"Colonne inconnue pour l'axe Y : {col!r}")
    elif y is not None and y not in columns:
        raise ValueError(f"Colonne inconnue pour l'axe Y : {y!r}")

    series = spec.get("series")
    if series is not None and series not in columns:
        raise ValueError(f"Colonne inconnue pour la série : {series!r}")

    group_by = spec.get("group_by")
    if group_by is not None and group_by not in columns:
        raise ValueError(f"Colonne inconnue pour le regroupement : {group_by!r}")

    aggregation = spec.get("aggregation", "none")
    if aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"Agrégation non supportée : {aggregation!r}")

    barmode = spec.get("barmode")
    if barmode is not None and barmode not in ALLOWED_BARMODES:
        raise ValueError(f"barmode non supporté : {barmode!r}")
    if chart_type == "bar" and series and not barmode:
        # Plotly's own default when a color/series grouping is added to a
        # bar chart is stacked, not side-by-side. "Clustered bar chart" is
        # the far more common ask when someone wants a category broken down
        # by a sub-category (e.g. customer x product), so that's the
        # default here; stacking is opt-in via an explicit "stack".
        spec["barmode"] = "group"

    if chart_type == "histogram":
        if x is None:
            raise ValueError("Un histogramme nécessite une colonne 'x'.")
    else:
        if x is None or y is None:
            raise ValueError(f"Le type de graphique {chart_type!r} nécessite 'x' et 'y'.")

    # Deterministic safety net: don't trust the model to always follow the
    # pie-chart category-count rule, enforce it ourselves.
    if chart_type == "pie" and x is not None:
        if df[x].nunique(dropna=True) > MAX_PIE_CATEGORIES:
            logger.info("Too many categories for a pie chart, falling back to bar.")
            spec["chart_type"] = "bar"

    return spec


def _prepare_data(spec: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a real pandas aggregation if the spec asked for one. This is the
    only place data is transformed, and it's plain deterministic pandas
    code operating on the real DataFrame, nothing the LLM typed.
    """
    aggregation = spec.get("aggregation", "none")
    group_by = spec.get("group_by")

    if aggregation == "none" or not group_by:
        return df

    y = spec.get("y")
    y_cols = y if isinstance(y, list) else [y]
    y_cols = [c for c in y_cols if c is not None]
    if not y_cols:
        return df

    agg_func = _AGG_FUNCS[aggregation]
    return df.groupby(group_by, as_index=False)[y_cols].agg(agg_func)


def _build_figure(spec: dict, df: pd.DataFrame):
    chart_type = spec["chart_type"]
    x = spec.get("x")
    y = spec.get("y")
    series = spec.get("series")
    title = spec.get("title") or ""
    x_label = spec.get("x_label") or x
    y_label = spec.get("y_label") or (y if isinstance(y, str) else None)

    labels = {}
    if x:
        labels[x] = x_label
    if isinstance(y, str) and y_label:
        labels[y] = y_label

    kwargs = {"title": title, "labels": labels}
    if series and chart_type != "pie":
        kwargs["color"] = series

    if chart_type == "bar":
        if series:
            kwargs["barmode"] = spec.get("barmode") or "group"
        fig = px.bar(df, x=x, y=y, **kwargs)
    elif chart_type == "line":
        fig = px.line(df, x=x, y=y, **kwargs)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, **kwargs)
    elif chart_type == "pie":
        values_col = y if isinstance(y, str) else y[0]
        fig = px.pie(df, names=x, values=values_col, title=title)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x, **kwargs)
    else:
        raise ValueError(f"Type de graphique non supporté : {chart_type!r}")

    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40))
    return fig


def is_chart_worthy(df: pd.DataFrame) -> bool:
    """
    Cheap, LLM-free heuristic used to decide (a) which tab a turn should
    default to and (b) whether the Graphique button should be enabled at
    all for that turn. A single-row scalar answer ("how many employees")
    has nothing to plot; anything with at least two rows and a mix of a
    numeric and a non-numeric column (category/date to plot against a
    value) generally does.
    """
    if df is None or df.empty or len(df) < 2:
        return False
    numeric_cols = df.select_dtypes(include="number").columns
    non_numeric_cols = df.select_dtypes(exclude="number").columns
    return len(numeric_cols) >= 1 and len(non_numeric_cols) >= 1


def evaluate_chart_spec(question: str, answer: str, spec: dict, chart_df: pd.DataFrame) -> dict:
    """
    Ask the model to sanity-check its own chart choice against the data it
    actually plotted. Note this reviews the spec and data, not the rendered
    image, this pipeline has no vision model, so it can't judge whether the
    chart looks good, only whether the type/column mapping seems sensible
    for the question.

    Same "fail open, single verdict word" design as evaluate_sql_result in
    nlp_to_sql.py, for the same reliability reasons.
    """
    preview = "(aucune donnée)" if chart_df.empty else chart_df.head(5).to_string(index=False)
    spec_text = json.dumps({k: v for k, v in spec.items() if v is not None}, ensure_ascii=False)

    prompt = f"""<|im_start|>system
Tu vérifies si le choix de graphique ci-dessous est une bonne façon de répondre visuellement à la question de l'utilisateur.
Tu peux réfléchir avant de répondre, mais termine toujours par une seule ligne :
"OK" si ce choix semble pertinent.
"RETRY: raison courte" si un autre type de graphique ou un autre choix de colonnes serait plus adapté.
Ta toute dernière ligne doit être exactement "OK" ou "RETRY: ...", rien d'autre après.
<|im_end|>
<|im_start|>user
Question : {question}
Réponse donnée à l'utilisateur : {answer}
Choix de graphique : {spec_text}
Aperçu des données tracées :
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
        logger.warning(f"Chart self-check call failed, defaulting to OK: {e}")
        return {"verdict": "OK", "reason": ""}

    last_line = raw.strip().splitlines()[-1].strip() if raw.strip() else ""
    upper = last_line.upper()
    if re.search(r"\bRETRY\b", upper):
        m = re.search(r"RETRY\b\s*:?\s*(.*)", last_line, re.IGNORECASE)
        reason = m.group(1).strip() if m else ""
        return {"verdict": "RETRY", "reason": reason or "Choix de graphique jugé peu adapté par le modèle."}
    if re.search(r"\bOK\b", upper):
        return {"verdict": "OK", "reason": ""}

    logger.warning(f"Could not parse chart self-check verdict, defaulting to OK: {raw!r}")
    return {"verdict": "OK", "reason": ""}


def generate_graph(question: str, answer: str, sql: str, df: pd.DataFrame,
                    max_attempts: int = None, schema: dict = None) -> dict:
    """
    Returns {"figure": <plotly figure as JSON-safe dict>, "chart_type": ..., "title": ...}
    on success, or {"error": "..."} with a user-facing French message if
    every attempt fails to even produce a valid, renderable spec.

    Each attempt: ask for a spec, validate + render it, then ask the model
    to self-check its own choice. A RETRY verdict feeds the reason back
    into the next attempt's prompt. On the final attempt, a rendered chart
    is returned even with a RETRY verdict (the model's own opinion isn't a
    guarantee, showing the best available chart beats showing nothing).

    schema, if provided (the app's SCHEMA, with schema_metadata.yaml
    enrichment already merged in), lets column descriptions reach the
    prompt on a best-effort basis, see _build_column_lookup.
    """
    if max_attempts is None:
        max_attempts = config.MAX_CHART_GENERATION_ATTEMPTS

    if df is None or df.empty:
        return {"error": "Aucune donnée disponible pour générer un graphique."}

    previous_spec = None
    previous_feedback = None
    last_result = None

    for attempt in range(1, max_attempts + 1):
        prompt = _build_prompt(question, answer, sql, df, previous_spec, previous_feedback, schema)
        raw_output = llm_utils.generate(
            prompt, max_tokens=config.REASONING_MAX_TOKENS, temperature=0,
            stop=["<|im_end|>"], role="reasoning"
        )

        try:
            spec = _extract_json(raw_output)
            spec = _validate_spec(spec, df)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Chart attempt {attempt}/{max_attempts}: spec rejected: {e} | raw: {raw_output!r}")
            previous_spec = {"attempted": True}
            previous_feedback = f"JSON invalide ou colonne inconnue : {e}"
            continue

        try:
            chart_df = _prepare_data(spec, df)
            fig = _build_figure(spec, chart_df)
        except Exception as e:
            logger.error(f"Chart attempt {attempt}/{max_attempts}: rendering failed: {e}")
            previous_spec = spec
            previous_feedback = f"Échec du rendu : {e}"
            continue

        last_result = {
            "figure": json.loads(fig.to_json()),
            "chart_type": spec["chart_type"],
            "title": spec.get("title", ""),
        }

        verdict = evaluate_chart_spec(question, answer, spec, chart_df)
        if verdict["verdict"] == "OK" or attempt == max_attempts:
            if verdict["verdict"] != "OK":
                logger.info(
                    f"Using chart despite a RETRY self-check after exhausting attempts: {verdict['reason']}"
                )
            return last_result

        logger.info(f"Chart attempt {attempt}/{max_attempts}: model flagged its own choice: {verdict['reason']}")
        previous_spec = spec
        previous_feedback = verdict["reason"]

    # Every attempt failed to even produce a valid, renderable spec.
    return {"error": "Impossible de déterminer un graphique adapté à ces données."}
