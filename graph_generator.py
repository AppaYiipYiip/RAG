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

import llm_utils

logger = logging.getLogger(__name__)

ALLOWED_CHART_TYPES = {"bar", "line", "scatter", "pie", "histogram"}
ALLOWED_AGGREGATIONS = {"sum", "avg", "count", "min", "max", "none"}

# Above this many distinct categories a pie chart stops being readable;
# we fall back to a bar chart automatically rather than trusting the model
# to always follow the instruction below.
MAX_PIE_CATEGORIES = 8

_AGG_FUNCS = {"sum": "sum", "avg": "mean", "count": "count", "min": "min", "max": "max"}

CHART_SPEC_DOC = """Tu dois répondre uniquement avec un objet JSON (rien avant, rien après) décrivant comment tracer un graphique à partir des données fournies. Format attendu :

{
  "chart_type": "bar" | "line" | "scatter" | "pie" | "histogram",
  "x": "nom_de_colonne",
  "y": "nom_de_colonne" ou ["colonne1", "colonne2"] pour plusieurs séries (null pour histogram),
  "series": "nom_de_colonne" ou null,
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
- Réponds uniquement avec le JSON, sans explication."""


def _build_prompt(question: str, answer: str, sql: str, df: pd.DataFrame) -> str:
    columns_block = "\n".join(f"  - {col} ({df[col].dtype})" for col in df.columns)
    preview = df.head(5).to_string(index=False)

    return f"""<|im_start|>system
Tu es un assistant qui choisit comment représenter graphiquement le résultat d'une requête SQL, pour aider un utilisateur à mieux comprendre la réponse à sa question.
{CHART_SPEC_DOC}
<|im_end|>
<|im_start|>user
Question de l'utilisateur : {question}

Réponse déjà donnée à l'utilisateur : {answer}

Colonnes disponibles dans les données :
{columns_block}

Aperçu des données (5 premières lignes) :
{preview}
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


def generate_graph(question: str, answer: str, sql: str, df: pd.DataFrame) -> dict:
    """
    Returns {"figure": <plotly figure as JSON-safe dict>, "chart_type": ..., "title": ...}
    on success, or {"error": "..."} with a user-facing French message on failure.
    Never raises for LLM-caused failures (bad JSON, unknown columns); those
    are caught and turned into a graceful error instead.
    """
    if df is None or df.empty:
        return {"error": "Aucune donnée disponible pour générer un graphique."}

    prompt = _build_prompt(question, answer, sql, df)
    raw_output = llm_utils.generate(prompt, max_tokens=300, temperature=0, stop=["<|im_end|>"])

    try:
        spec = _extract_json(raw_output)
        spec = _validate_spec(spec, df)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Chart spec rejected: {e} | raw output: {raw_output!r}")
        return {"error": "Impossible de déterminer un graphique adapté à ces données."}

    try:
        chart_df = _prepare_data(spec, df)
        fig = _build_figure(spec, chart_df)
    except Exception as e:
        logger.error(f"Chart rendering failed: {e}")
        return {"error": "La génération du graphique a échoué."}

    figure_json = json.loads(fig.to_json())

    return {
        "figure": figure_json,
        "chart_type": spec["chart_type"],
        "title": spec.get("title", ""),
    }
