"""
tasks.py — Three SQL optimization tasks for SQLOptimEnv
────────────────────────────────────────────────────────
Easy   : fix a syntax error in a simple SELECT
Medium : rewrite a correlated subquery as an efficient JOIN
Hard   : optimize a multi-table query with redundant clauses + suboptimal joins
"""

import sqlite3
import traceback
from typing import Any

# Import helpers from environment (lazy to avoid circular)
# Graders receive `con` directly and call run_query themselves.


# ─── Shared seed data ────────────────────────────────────────────────────────

_ECOMMERCE_SETUP = [
    """
    CREATE TABLE IF NOT EXISTS customers (
        id      INTEGER PRIMARY KEY,
        name    TEXT NOT NULL,
        city    TEXT NOT NULL,
        tier    TEXT NOT NULL DEFAULT 'standard'  -- 'standard' | 'premium'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        id          INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL REFERENCES customers(id),
        amount      REAL    NOT NULL,
        status      TEXT    NOT NULL,  -- 'completed' | 'pending' | 'cancelled'
        created_at  TEXT    NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS products (
        id    INTEGER PRIMARY KEY,
        name  TEXT NOT NULL,
        price REAL NOT NULL,
        category TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS order_items (
        id         INTEGER PRIMARY KEY,
        order_id   INTEGER NOT NULL REFERENCES orders(id),
        product_id INTEGER NOT NULL REFERENCES products(id),
        qty        INTEGER NOT NULL,
        unit_price REAL    NOT NULL
    );
    """,
    # Seed customers
    """
    INSERT INTO customers VALUES
      (1,'Alice','Mumbai','premium'),
      (2,'Bob','Delhi','standard'),
      (3,'Carol','Pune','premium'),
      (4,'Dave','Chennai','standard'),
      (5,'Eve','Bangalore','premium');
    """,
    # Seed products
    """
    INSERT INTO products VALUES
      (1,'Laptop',75000,'electronics'),
      (2,'Phone',25000,'electronics'),
      (3,'Desk',8000,'furniture'),
      (4,'Chair',4500,'furniture'),
      (5,'Headphones',3000,'electronics');
    """,
    # Seed orders (100 rows via recursive CTE)
    """
    WITH RECURSIVE gen(n) AS (
      SELECT 1 UNION ALL SELECT n+1 FROM gen WHERE n < 100
    )
    INSERT INTO orders(id, customer_id, amount, status, created_at)
    SELECT
      n,
      ((n-1) % 5) + 1,
      ROUND(1000 + (n * 317.7) % 90000, 2),
      CASE (n % 3) WHEN 0 THEN 'completed' WHEN 1 THEN 'pending' ELSE 'cancelled' END,
      DATE('2024-01-01', '+' || (n*3) || ' days')
    FROM gen;
    """,
    # Seed order_items (300 rows)
    """
    WITH RECURSIVE gen(n) AS (
      SELECT 1 UNION ALL SELECT n+1 FROM gen WHERE n < 300
    )
    INSERT INTO order_items(id, order_id, product_id, qty, unit_price)
    SELECT
      n,
      ((n-1) % 100) + 1,
      ((n-1) % 5) + 1,
      (n % 5) + 1,
      ROUND(500 + (n * 137.3) % 70000, 2)
    FROM gen;
    """,
]

_SCHEMA_DOC = """
customers(id, name, city, tier)
orders(id, customer_id, amount, status, created_at)
products(id, name, price, category)
order_items(id, order_id, product_id, qty, unit_price)
"""


# ─── Grader helpers ──────────────────────────────────────────────────────────

def _run(con: sqlite3.Connection, sql: str, iters: int = 40):
    """Returns (rows_as_dicts, avg_ms). Raises on error."""
    import time
    cur = con.cursor()
    cur.execute(sql)
    rows = [dict(r) for r in cur.fetchall()]
    t0 = time.perf_counter()
    for _ in range(iters):
        cur.execute(sql)
        cur.fetchall()
    ms = (time.perf_counter() - t0) / iters * 1000
    return rows, ms


def _rows_equal(a, b) -> bool:
    """Order-insensitive row comparison."""
    def norm(rows):
        return sorted([tuple(sorted(r.items())) for r in rows])
    return norm(a) == norm(b)


# ─── Task definitions ────────────────────────────────────────────────────────

def _grade_easy(con, original_query, rewritten_query):
    
    # Correct answer: fix the syntax error — query must run and return same rows as reference
    reference_sql = """
        SELECT c.name, COUNT(o.id) AS order_count
        FROM customers c
        JOIN orders o ON o.customer_id = c.id
        WHERE o.status = 'completed'
        GROUP BY c.id, c.name
        ORDER BY order_count DESC;
    """
    try:
        ref_rows, _ = _run(con, reference_sql)
    except Exception as e:
        return dict(score=0.0, correct=False, faster=False,
                      error=f"Reference query failed: {e}")

    try:
        rew_rows, rew_ms = _run(con, rewritten_query)
    except Exception as e:
        return dict(score=0.0, correct=False, faster=False,
                      error=f"Rewritten query error: {e}")

    if not _rows_equal(ref_rows, rew_rows):
        return dict(score=0.3, correct=False, faster=False,
                      error="Query runs but returns wrong results.")

    return dict(score=1.0, correct=True, faster=True,
                  error=None, speedup=1.0)


def _grade_medium(con, original_query, rewritten_query):
    
    # Original uses correlated subquery; good answer uses JOIN — measure both
    reference_sql = """
        SELECT c.name, c.city,
               SUM(o.amount) AS total_spent
        FROM customers c
        JOIN orders o ON o.customer_id = c.id
        WHERE o.status = 'completed'
        GROUP BY c.id, c.name, c.city
        HAVING total_spent > 50000
        ORDER BY total_spent DESC;
    """
    try:
        ref_rows, _ = _run(con, reference_sql)
        orig_rows, orig_ms = _run(con, original_query)
    except Exception as e:
        return dict(score=0.0, correct=False, faster=False,
                      error=f"Setup error: {e}")

    try:
        rew_rows, rew_ms = _run(con, rewritten_query)
    except Exception as e:
        return dict(score=0.0, correct=False, faster=False,
                      error=f"Rewritten query error: {e}")

    correct = _rows_equal(ref_rows, rew_rows)
    if not correct:
        # Partial credit if it at least runs without error
        return dict(score=0.2, correct=False, faster=False,
                      error="Query runs but returns wrong results.")

    speedup = orig_ms / rew_ms if rew_ms > 0 else 1.0
    faster = speedup >= 1.1  # at least 10% faster

    if correct and faster:
        score = min(1.0, 0.7 + 0.3 * min(speedup / 3.0, 1.0))
    else:
        score = 0.65  # correct but not faster — still decent

    return dict(score=round(score, 3), correct=correct,
                  faster=faster, speedup=round(speedup, 3))


def _grade_hard(con, original_query, rewritten_query):
    
    reference_sql = """
        SELECT
            p.category,
            p.name        AS product_name,
            SUM(oi.qty * oi.unit_price) AS revenue,
            COUNT(DISTINCT o.customer_id) AS unique_buyers
        FROM order_items oi
        JOIN orders   o ON o.id  = oi.order_id   AND o.status = 'completed'
        JOIN products p ON p.id  = oi.product_id
        JOIN customers c ON c.id = o.customer_id AND c.tier = 'premium'
        GROUP BY p.category, p.id, p.name
        ORDER BY revenue DESC;
    """
    try:
        ref_rows, _ = _run(con, reference_sql)
        orig_rows, orig_ms = _run(con, original_query)
    except Exception as e:
        return dict(score=0.0, correct=False, faster=False,
                      error=f"Setup error: {e}")

    try:
        rew_rows, rew_ms = _run(con, rewritten_query)
    except Exception as e:
        return dict(score=0.0, correct=False, faster=False,
                      error=f"Rewritten query error: {e}")

    correct = _rows_equal(ref_rows, rew_rows)
    if not correct:
        return dict(score=0.15, correct=False, faster=False,
                      error="Query runs but returns wrong results.")

    speedup = orig_ms / rew_ms if rew_ms > 0 else 1.0
    faster = speedup >= 1.1

    if correct and faster:
        score = min(1.0, 0.6 + 0.4 * min(speedup / 4.0, 1.0))
    else:
        score = 0.6

    return dict(score=round(score, 3), correct=correct,
                  faster=faster, speedup=round(speedup, 3))


# ─── TASKS registry ──────────────────────────────────────────────────────────

TASKS: dict[str, dict] = {

    # ── EASY ─────────────────────────────────────────────────────────────────
    "easy_syntax_fix": {
        "description": (
            "EASY — Fix the syntax error in this SQL query so it runs correctly "
            "and returns the number of completed orders per customer, ordered by count descending. "
            "The schema is an e-commerce database."
        ),
        "schema_sql": _SCHEMA_DOC,
        "setup_sql": _ECOMMERCE_SETUP,
        "original_query": """
            SELECT c.name, COUNT(o.id) AS order_count
            FORM customers c                          -- typo: FORM instead of FROM
            JOIN orders o ON o.customer_id = c.id
            WHERE o.status = 'completed'
            GROUP BY c.id c.name                      -- missing comma
            ORDER BY order_count DESC;
        """,
        "hint": "There are two syntax errors. Look at the FROM clause and the GROUP BY clause.",
        "grader": _grade_easy,
    },

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    "medium_subquery_to_join": {
        "description": (
            "MEDIUM — The query below uses a correlated subquery to find customers who have "
            "spent more than 50,000 on completed orders. Rewrite it using a JOIN + GROUP BY + HAVING "
            "so it produces the same result but runs faster."
        ),
        "schema_sql": _SCHEMA_DOC,
        "setup_sql": _ECOMMERCE_SETUP,
        "original_query": """
            SELECT c.name, c.city,
                   (SELECT SUM(o.amount)
                    FROM orders o
                    WHERE o.customer_id = c.id
                      AND o.status = 'completed') AS total_spent
            FROM customers c
            WHERE (
                SELECT SUM(o.amount)
                FROM orders o
                WHERE o.customer_id = c.id
                  AND o.status = 'completed'
            ) > 50000
            ORDER BY total_spent DESC;
        """,
        "hint": "Replace the two correlated subqueries with a single JOIN + GROUP BY + HAVING.",
        "grader": _grade_medium,
    },

    # ── HARD ──────────────────────────────────────────────────────────────────
    "hard_multi_table_optimize": {
        "description": (
            "HARD — This query calculates revenue and unique buyers per product for premium customers "
            "with completed orders. It is bloated with redundant subqueries and Cartesian-join risks. "
            "Rewrite it to be correct AND faster using clean JOIN syntax with proper filter placement."
        ),
        "schema_sql": _SCHEMA_DOC,
        "setup_sql": _ECOMMERCE_SETUP,
        "original_query": """
            SELECT
                p.category,
                p.name AS product_name,
                SUM(oi.qty * oi.unit_price) AS revenue,
                COUNT(DISTINCT o.customer_id) AS unique_buyers
            FROM order_items oi, orders o, products p, customers c
            WHERE oi.order_id = o.id
              AND oi.product_id = p.id
              AND o.customer_id = c.id
              AND o.status = (SELECT 'completed')        -- pointless scalar subquery
              AND c.tier = (SELECT 'premium' FROM customers LIMIT 1)  -- redundant subquery
              AND p.id IN (SELECT id FROM products)      -- redundant filter
            GROUP BY p.category, p.id, p.name
            ORDER BY revenue DESC;
        """,
        "hint": (
            "Replace implicit Cartesian joins (comma syntax) with explicit JOIN … ON. "
            "Remove the three pointless scalar subqueries and inline the filter literals directly. "
            "Move filter conditions into JOIN ON clauses where possible."
        ),
        "grader": _grade_hard,
    },
}
