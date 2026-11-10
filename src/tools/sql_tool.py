
"""Read-only SQL analytics tool for the enterprise platform."""

import logging
import sqlite3
from pathlib import Path

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = PROJECT_ROOT / "data" / "analytics" / "sales.csv"
DB_FILE = PROJECT_ROOT / "data" / "analytics" / "sales.db"

logger = logging.getLogger(__name__)


def initialize_database() -> None:
    """Load the sample CSV into a local SQLite database."""
    import pandas as pd

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Sales CSV not found: {CSV_FILE}"
        )

    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(CSV_FILE)

    required_columns = {
        "order_id",
        "region",
        "product",
        "category",
        "quantity",
        "unit_price",
    }

    if not required_columns.issubset(sales.columns):
        raise ValueError(
            f"CSV must contain columns: {sorted(required_columns)}"
        )

    if sales.empty:
        raise ValueError("The sales CSV contains no records.")

    if sales[list(required_columns)].isnull().any().any():
        raise ValueError("The sales CSV contains missing values.")

    if (sales["quantity"] < 0).any() or (sales["unit_price"] < 0).any():
        raise ValueError("Quantity and unit_price must be non-negative.")

    with sqlite3.connect(DB_FILE) as connection:
        sales.to_sql("sales", connection, if_exists="replace", index=False)

    logger.info(
        "Initialized SQLite database with %d sales records",
        len(sales),
    )


@tool
def query_sales_database(sql_query: str) -> str:
    """Run a read-only SQL SELECT query against the sales table.

    The table contains order_id, region, product, category, quantity,
    and unit_price. Use this tool for aggregations, comparisons, totals,
    averages, and other questions about the sample sales data.
    Revenue can be calculated as SUM(quantity * unit_price).

    Only a single SELECT query or a WITH ... SELECT query is permitted.
    """
    query = sql_query.strip()

    if not query:
        return "Error: SQL query cannot be empty."

    normalized = query.lower()

    if not (normalized.startswith("select") or normalized.startswith("with")):
        return "Error: Only SELECT queries are allowed."

    if ";" in query.rstrip(";"):
        return "Error: Only one SQL statement is allowed."

    if "--" in query or "/*" in query or "*/" in query:
        return "Error: SQL comments are not allowed."

    if not DB_FILE.exists():
        return "Error: Database not found. Initialize it first."

    try:
        # Open the existing database in SQLite read-only mode.
        connection = sqlite3.connect(
            f"{DB_FILE.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=5,
        )

        try:
            connection.set_authorizer(sqlite_authorizer)
            connection.set_progress_handler(
                lambda: 1, 100_000
            )

            cursor = connection.execute(query)
            rows = cursor.fetchmany(101)

            if len(rows) > 100:
                return "Error: Query returned more than 100 rows."

            columns = [
                description[0]
                for description in cursor.description or []
            ]

            if not columns:
                return "Error: The query did not return a result set."

            if not rows:
                return "Query completed successfully. No rows returned."

            output = [" | ".join(columns)]
            output.append("-" * 60)

            for row in rows:
                output.append(
                    " | ".join(str(value) for value in row)
                )

            return "\n".join(output)

        finally:
            connection.close()

    except sqlite3.Error:
        logger.exception("SQL query failed")
        return "Error: The SQL query could not be executed. Check its syntax and table names."


def sqlite_authorizer(action, arg1, arg2, database_name, trigger_name):
    """Permit reads and safe query operations, but block database changes."""
    allowed_actions = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }

    if action in allowed_actions:
        return sqlite3.SQLITE_OK

    return sqlite3.SQLITE_DENY


def main() -> None:
    initialize_database()

    queries = [
        (
            "Total revenue by region",
            """
            SELECT region, SUM(quantity * unit_price) AS revenue
            FROM sales
            GROUP BY region
            ORDER BY revenue DESC
            """,
        ),
        (
            "Revenue by category",
            """
            SELECT category, SUM(quantity * unit_price) AS revenue
            FROM sales
            GROUP BY category
            ORDER BY revenue DESC
            """,
        ),
    ]

    for label, query in queries:
        print(f"\n{'=' * 60}")
        print(label)
        print("=" * 60)
        print(query_sales_database.invoke({"sql_query": query}))


if __name__ == "__main__":
    main()
