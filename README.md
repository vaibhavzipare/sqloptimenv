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

## Tasks

- 🟢 **easy_syntax_fix** — Fix two syntax errors in a simple GROUP BY query
- 🟡 **medium_subquery_to_join** — Replace correlated subquery with JOIN + HAVING
- 🔴 **hard_multi_table_optimize** — Eliminate redundant subqueries, rewrite Cartesian joins

## Reward Function

| Outcome | Score |
|---|---|
| Query errors out | 0.0 |
| Runs but wrong results | 0.15–0.30 |
| Correct, not faster | 0.60–0.65 |
| Correct + faster | 0.70–1.00 |

## Setup

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Baseline Scores

| Task | Score |
|---|---|
| easy_syntax_fix | ~1.00 |
| medium_subquery_to_join | ~0.75 |
| hard_multi_table_optimize | ~0.65 |
