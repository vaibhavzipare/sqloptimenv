"""
inference.py — Baseline inference script for SQLOptimEnv
=========================================================
Mandatory stdout format:
  [START] task=<name> env=sqloptimenv model=<model>
  [STEP]  step=<n> action=<sql> reward=<0.00> done=<bool> error=<msg|null>
  [END]   success=<bool> steps=<n> score=<0.00> rewards=<r1,r2,...>

Environment variables:
  API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model id       (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key
"""

import os
import sys
import json
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]  # STRICT (no fallback)
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.environ["API_KEY"]  # STRICT (no fallback)
ENV_URL      = (os.getenv("ENV_BASE_URL") or os.getenv("ENV_URL") or "http://localhost:7860").rstrip("/")
BENCHMARK    = "sqloptimenv"
MAX_STEPS    = 5

TASKS = ["easy_syntax_fix", "medium_subquery_to_join", "hard_multi_table_optimize"]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ── Env HTTP helpers ──────────────────────────────────────────────────────────

def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(rewritten_query: str) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"rewritten_query": rewritten_query}, timeout=30)
    r.raise_for_status()
    return r.json()


# ── LLM call ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert SQL query optimizer.
You will receive a broken or slow SQL query along with its schema.
Your job: rewrite the query to fix errors and/or improve performance.
Reply with ONLY the corrected SQL query — no explanation, no markdown fences, just raw SQL."""
def get_rewrite(obs: dict) -> str:
    parts = [
        f"Schema:\n{obs['schema_sql']}",
        f"Original query:\n{obs['original_query']}",
        f"Task description:\n{obs['description']}",
    ]
    if obs.get("hint"):
        parts.append(f"Hint: {obs['hint']}")
    if obs.get("last_error"):
        parts.append(f"Previous attempt error: {obs['last_error']}")

    user_msg = "\n\n".join(parts)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()

# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str) -> float:
    obs = env_reset(task_id)
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards = []
    step_n  = 0
    score   = 0.0
    success = False

    for _ in range(MAX_STEPS):
        step_n += 1
        rewrite = get_rewrite(obs)

        # Truncate for log line (no newlines allowed)
        action_log = rewrite.replace("\n", " ").replace("\r", "")[:120]

        result = env_step(rewrite)
        reward = result["reward"]
        done   = result["done"]
        info   = result.get("info", {})
        error  = info.get("error") or "null"
        if error and error != "null":
            error = error.replace("\n", " ")[:80]

        rewards.append(reward)
        obs = result["observation"]

        print(
            f"[STEP] step={step_n} action={action_log} "
            f"reward={reward:.2f} done={str(done).lower()} error={error}",
            flush=True,
        )

        if done:
            score   = reward
            success = info.get("correct", False)
            break

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_n} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )
    return score


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    task_arg = os.getenv("TASK_ID")
    tasks_to_run = [task_arg] if task_arg else TASKS

    all_scores = {}
    for task_id in tasks_to_run:
        print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)
        try:
            s = run_episode(task_id)
        except Exception as e:
            print(f"[DEBUG] Exception in task {task_id}: {e}", flush=True)
            print(f"[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)
            s = 0.0
        all_scores[task_id] = s
        print("", flush=True)

    print("=== Baseline Results ===")
    for t, s in all_scores.items():
        print(f"  {t}: {s:.3f}")
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  AVERAGE: {avg:.3f}")
