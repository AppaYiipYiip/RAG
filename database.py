import re
import pyodbc
import pandas as pd
from config import DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD

# Keywords that indicate a write, DDL, or admin statement. Blocked outright.
_DISALLOWED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|EXEC|EXECUTE|MERGE|CREATE|"
    r"GRANT|REVOKE|DENY|BACKUP|RESTORE|sp_|xp_)\b",
    re.IGNORECASE,
)


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
        rows = cursor.fetchall()
        df = pd.DataFrame.from_records(rows, columns=columns)
        return df
    finally:
        conn.close()