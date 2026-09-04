# schema_loader.py
import logging
import os

import pyodbc
import yaml

from config import DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD, SCHEMA_METADATA_PATH

logger = logging.getLogger(__name__)

def get_connection():
    conn_str = (
        f'DRIVER={{ODBC Driver 17 for SQL Server}};'
        f'SERVER={DB_SERVER};'
        f'DATABASE={DB_NAME};'
        f'UID={DB_USER};'
        f'PWD={DB_PASSWORD};'
    )
    return pyodbc.connect(conn_str)

def get_schema():
    """Query the database and return a dictionary of tables, columns, and foreign keys."""
    conn = get_connection()
    cursor = conn.cursor()

    schema = {"tables": []}

    # Get all tables
    cursor.execute("""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
          AND TABLE_NAME NOT LIKE 'sys%'
        ORDER BY TABLE_NAME
    """)
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        table_info = {"name": table, "columns": [], "foreign_keys": []}

        # Get columns
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE,
                   COLUMNPROPERTY(OBJECT_ID(TABLE_NAME), COLUMN_NAME, 'IsIdentity') AS IsIdentity,
                   COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
        """, table)
        for col in cursor.fetchall():
            col_name, data_type, is_nullable, is_identity, default = col
            table_info["columns"].append({
                "name": col_name,
                "type": data_type,
                "nullable": is_nullable == "YES",
                "primary_key": False,  # will update later
                "identity": bool(is_identity)
            })

        # Get primary key columns
        cursor.execute("""
            SELECT KU.COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS TC
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS KU
                ON TC.CONSTRAINT_NAME = KU.CONSTRAINT_NAME
            WHERE TC.CONSTRAINT_TYPE = 'PRIMARY KEY'
              AND TC.TABLE_NAME = ?
        """, table)
        pk_cols = [row[0] for row in cursor.fetchall()]
        for col in table_info["columns"]:
            if col["name"] in pk_cols:
                col["primary_key"] = True

        # Get foreign keys
        cursor.execute("""
            SELECT
                fk.name AS FK_name,
                COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS FK_column,
                OBJECT_NAME(fkc.referenced_object_id) AS Referenced_table,
                COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS Referenced_column
            FROM sys.foreign_keys AS fk
            JOIN sys.foreign_key_columns AS fkc ON fk.object_id = fkc.constraint_object_id
            WHERE OBJECT_NAME(fkc.parent_object_id) = ?
        """, table)
        for fk in cursor.fetchall():
            table_info["foreign_keys"].append({
                "column": fk[1],
                "references_table": fk[2],
                "references_column": fk[3]
            })

        schema["tables"].append(table_info)

    conn.close()
    return schema


def load_schema_metadata(path: str = None) -> dict:
    """
    Load an optional, hand-written enrichment file layering business
    context on top of the auto-introspected schema: what a table/column
    actually means, what it's called in conversation, sample values. See
    schema_metadata.example.yaml for the format and a starter you can copy.

    This is the only piece of schema knowledge that has to come from a
    human, the structural facts (names, types, keys) are always
    re-introspected fresh from the live database on every startup and can
    never go stale, but no amount of introspection can tell us that
    "poste" means job title or that a self-joined "nom_2" is the second
    employee's name. Returns {} (schema works fine without any enrichment)
    if the file doesn't exist.
    """
    if path is None:
        path = SCHEMA_METADATA_PATH
    if not os.path.exists(path):
        logger.info(f"No schema metadata file at {path!r}, using the auto-introspected schema as-is.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f) or {}
    table_count = len(metadata.get("tables", {}))
    logger.info(f"Loaded schema metadata from {path!r}: {table_count} table(s) enriched.")
    return metadata


def merge_schema_metadata(schema: dict, metadata: dict) -> dict:
    """
    Layer descriptions/synonyms/examples from metadata onto the
    auto-introspected schema, in place, and also return it. Purely
    additive: any table or column not mentioned in metadata is left
    exactly as introspected, and a table/column that no longer exists in
    the metadata file but did exist in the schema is simply not enriched,
    it's never removed by this.
    """
    tables_meta = metadata.get("tables", {})
    for table in schema.get("tables", []):
        table_meta = tables_meta.get(table["name"])
        if not table_meta:
            continue
        if table_meta.get("description"):
            table["description"] = table_meta["description"]
        columns_meta = table_meta.get("columns", {})
        for col in table.get("columns", []):
            col_meta = columns_meta.get(col["name"])
            if not col_meta:
                continue
            if col_meta.get("description"):
                col["description"] = col_meta["description"]
            if col_meta.get("synonyms"):
                col["synonyms"] = col_meta["synonyms"]
            if "example" in col_meta:
                col["example"] = col_meta["example"]
    return schema