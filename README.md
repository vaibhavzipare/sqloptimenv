---
title: SQLOptimEnv
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# SQLOptimEnv 🗄️⚡

> An **OpenEnv**-compliant RL environment where an AI agent receives broken or slow SQL queries and must rewrite them to be correct and performant.

---

## Overview

SQLOptimEnv simulates a real-world task that database engineers face daily: **fixing broken queries and optimizing slow ones**. The agent receives a query + schema, rewrites it, and a deterministic grader evaluates correctness (same result set) and performance (execution speed).

This is a genuine, practically useful benchmark — real applications break on bad SQL, and LLM-powered SQL assistance is an active research and product area.

---

## Environment Design

### Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Active task identifier |
| `description` | string | Natural language task description |
| `schema_sql` | string | Table definitions |
| `original_query` | string | The broken/slow query to fix |
| `hint` | string \| null | Optional hint for the agent |
| `step` | int | Current step number |
| `last_error` | string \| null | Error from previous attempt |
| `last_reward` | float | Reward from previous step |

### Action Space

| Field | Type | Description |
|---|---|---|
| `rewritten_query` | string | Agent's rewritten SQL query |

### Reward Function

| Outcome | Score |
|---|---|
| Query errors out | 0.0 |
| Runs but wrong results | 0.15 – 0.30 |
| Correct, not faster | 0.60 – 0.65 |
| Correct + faster (≥10%) | 0.70 – 1.00 (scales with speedup) |

Rewards provide **partial progress signal** at every step, not just binary end-of-episode.

---

## Tasks

### 🟢 Easy — `easy_syntax_fix`
Fix two syntax errors (`FORM` instead of `FROM`, missing comma in `GROUP BY`) in a simple aggregation query. Tests basic SQL repair.

**Expected difficulty**: Frontier models score ~1.0

### 🟡 Medium — `medium_subquery_to_join`
A correlated subquery runs a full scan per customer row. Rewrite using `JOIN + GROUP BY + HAVING` for the same results but faster. Tests query restructuring.

**Expected difficulty**: Strong models score 0.7–1.0

### 🔴 Hard — `hard_multi_table_optimize`
A four-table query using implicit Cartesian join syntax with three redundant scalar subqueries. Rewrite using explicit `JOIN … ON` with filter conditions pushed into joins.

**Expected difficulty**: Even frontier models may score 0.6–0.85

---

## Setup & Usage

### Local

```bash
git clone <repo>
cd sqloptimenv
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
```

### Docker

```bash
docker build -t sqloptimenv .
docker run -p 7860:7860 sqloptimenv
```

### API

```bash
# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_syntax_fix"}'

# Submit a rewrite
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"rewritten_query": "SELECT c.name, COUNT(o.id) AS order_count FROM customers c JOIN orders o ON o.customer_id = c.id WHERE o.status = '\''completed'\'' GROUP BY c.id, c.name ORDER BY order_count DESC;"}'

# Get current state
curl http://localhost:7860/state
```

### Run inference baseline

```bash
export HF_TOKEN=your_token
export ENV_URL=http://localhost:7860
python inference.py
```

---

## Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Score |
|---|---|
| easy_syntax_fix | ~1.00 |
| medium_subquery_to_join | ~0.75 |
| hard_multi_table_optimize | ~0.65 |
| **Average** | **~0.80** |

---

## Project Structure

```
sqloptimenv/
├── app.py            # FastAPI server (OpenEnv HTTP interface)
├── environment.py    # Core env: Observation, Action, Reward, SQLOptimEnv
├── tasks.py          # 3 tasks + deterministic graders
├── inference.py      # Baseline inference script (mandatory)
├── openenv.yaml      # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```
