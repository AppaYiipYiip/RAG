import logging
import re
import pyodbc
import pandas as pd
from config import DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD

logger = logging.getLogger(__name__)

# Keywords that indicate a write, DDL, or admin statement. Blocked outright.
_DISALLOWED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|EXEC|EXECUTE|MERGE|CREATE|"
    r"GRANT|REVOKE|DENY|BACKUP|RESTORE|sp_|xp_)\b",
    re.IGNORECASE,
)


def _deduplicate_columns(columns):
    """
    SQL results can contain duplicate column names, most commonly from an
    unaliased self-join (SELECT e1.nom, e2.nom FROM employees e1 JOIN
    employees e2 ...). pandas allows duplicate DataFrame column labels, but
    df[col] then returns a DataFrame instead of a Series for that label,
    which silently breaks any code assuming a single Series (e.g. reading
    .dtype). Renaming duplicates up front means every downstream consumer
    (graph_generator, answer_generator, the frontend table) can safely
    assume every column name is unique.
    """
    seen = {}
    result = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 1
            result.append(col)
    return result


def is_read_only_query(sql: str) -> bool:
    """
    Return True only if sql looks like a single, read-only SELECT statement
    (optionally preceded by a WITH clause for CTEs).

    This is a defense-in-depth check on LLM-generated SQL, not a substitute
    for running the app under a database account that only has SELECT
    permissions.
    """
    if not sql or not sql.strip():
        return False

    # Strip comments so keywords can't be hidden inside them.
    stripped = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL).strip()

    if not stripped:
        return False

    # Allow one optional trailing semicolon, but reject stacked statements.
    body = stripped[:-1] if stripped.endswith(";") else stripped
    if ";" in body:
        return False

    # Must open with SELECT, or WITH ... for a common table expression.
    if not re.match(r"^\s*(SELECT|WITH)\b", body, re.IGNORECASE):
        return False

    # SELECT ... INTO creates a table, so treat it as a write.
    if re.search(r"\bSELECT\b.*\bINTO\b", body, re.IGNORECASE | re.DOTALL):
        return False

    if _DISALLOWED_KEYWORDS.search(body):
        return False

    return True


def get_connection():
    conn_str = (
        f'DRIVER={{ODBC Driver 17 for SQL Server}};'
        f'SERVER={DB_SERVER};'
        f'DATABASE={DB_NAME};'
        f'UID={DB_USER};'
        f'PWD={DB_PASSWORD};'
    )
    return pyodbc.connect(conn_str)

def execute_sql(sql_query: str) -> pd.DataFrame:
    """Execute a read-only SQL query and return results as a pandas DataFrame."""
    if not is_read_only_query(sql_query):
        raise ValueError(
            "Requête refusée : seules les requêtes SELECT en lecture seule sont autorisées."
        )

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        columns = [col[0] for col in cursor.description] if cursor.description else []
        deduped_columns = _deduplicate_columns(columns)
        if deduped_columns != columns:
            logger.warning(f"Query result had duplicate column names, renamed: {columns} -> {deduped_columns}")
        rows = cursor.fetchall()
        df = pd.DataFrame.from_records(rows, columns=deduped_columns)
        logger.debug(f"Result shape: {df.shape[0]} rows x {df.shape[1]} cols | dtypes: {df.dtypes.to_dict()}")
        return df
    finally:
        conn.close()