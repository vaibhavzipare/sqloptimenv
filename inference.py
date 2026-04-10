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
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "sk-no-key"
ENV_URL = os.getenv("ENV_BASE_URL") or os.getenv("ENV_URL", "http://localhost:7860")
ENV_URL = ENV_URL.rstrip("/")
BENCHMARK    = "sqloptimenv"
MAX_STEPS    = 5

TASKS = ["easy_syntax_fix", "medium_subquery_to_join", "hard_multi_table_optimize"]



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
     client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
     user_msg = f"""Schema:
{obs['schema_sql']}

Original query:
{obs['original_query']}

Task description:
{obs['description']}
"""
    if obs.get("hint"):
        user_msg += f"\nHint: {obs['hint']}"
    if obs.get("last_error"):
        user_msg += f"\nPrevious attempt error: {obs['last_error']}"

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
        try:
            s = run_episode(task_id)
        except Exception as e:
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
            s = 0.0
        all_scores[task_id] = s
        print("", flush=True)  # blank line between tasks

    print("=== Baseline Results ===")
    for t, s in all_scores.items():
        print(f"  {t}: {s:.3f}")
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  AVERAGE: {avg:.3f}")
