"""
SQLOptimEnv — OpenEnv-compliant SQL Query Optimization Environment
An AI agent receives broken or slow SQL queries and must rewrite them.
Graders evaluate correctness and performance improvement.
"""

import sqlite3
import time
import traceback
from typing import Any, Optional

from pydantic import BaseModel

from tasks import TASKS


# ── Typed Models ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    description: str
    schema_sql: str
    original_query: str
    hint: Optional[str] = None
    step: int = 0
    last_error: Optional[str] = None
    last_reward: float = 0.0


class Action(BaseModel):
    rewritten_query: str


class Reward(BaseModel):
    score: float          # 0.0 – 1.0
    correct: bool
    faster: bool
    error: Optional[str] = None
    speedup: float = 0.0  # ratio: original_time / rewritten_time


class State(BaseModel):
    task_id: str
    step: int
    done: bool
    last_reward: float
    last_error: Optional[str]


# ── Environment ───────────────────────────────────────────────────────────────

class SQLOptimEnv:
    """
    OpenEnv-compliant environment for SQL query optimization.

    Episode flow:
        obs = env.reset(task_id)          # choose a task
        obs, reward, done, info = env.step(action)
        state = env.state()
    """

    MAX_STEPS = 5

    def __init__(self, task_id: str = "easy_syntax_fix"):
        self._task_id = task_id
        self._step = 0
        self._done = False
        self._last_reward: float = 0.0
        self._last_error: Optional[str] = None
        self._con: Optional[sqlite3.Connection] = None
        self._task = TASKS[task_id]

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id:
            self._task_id = task_id
            self._task = TASKS[task_id]

        self._step = 0
        self._done = False
        self._last_reward = 0.0
        self._last_error = None

        # Fresh in-memory DB for every episode
        if self._con:
            self._con.close()
        self._con = sqlite3.connect(":memory:")
        self._con.row_factory = sqlite3.Row

        # Populate schema + seed data
        for stmt in self._task["setup_sql"]:
            self._con.executescript(stmt)
        self._con.commit()

        return self._build_obs()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done — call reset() first.")

        self._step += 1
        reward_obj = self._grade(action.rewritten_query)

        self._last_reward = reward_obj.score
        self._last_error = reward_obj.error

        # Episode ends on correct answer OR max steps reached
        if reward_obj.correct or self._step >= self.MAX_STEPS:
            self._done = True

        obs = self._build_obs()
        info = reward_obj.model_dump()
        return obs, reward_obj.score, self._done, info

    def state(self) -> State:
        return State(
            task_id=self._task_id,
            step=self._step,
            done=self._done,
            last_reward=self._last_reward,
            last_error=self._last_error,
        )

    def close(self):
        if self._con:
            self._con.close()
            self._con = None

    # ── Internal ───────────────────────────────────────────────────────────

    def _build_obs(self) -> Observation:
        return Observation(
            task_id=self._task_id,
            description=self._task["description"],
            schema_sql=self._task["schema_sql"],
            original_query=self._task["original_query"],
            hint=self._task.get("hint"),
            step=self._step,
            last_error=self._last_error,
            last_reward=self._last_reward,
        )

    def _grade(self, rewritten_query: str) -> Reward:
        grader = self._task["grader"]
        try:
            return grader(self._con, self._task["original_query"], rewritten_query)
        except Exception as e:
            return Reward(
                score=0.0,
                correct=False,
                faster=False,
                error=f"Grader exception: {traceback.format_exc(limit=2)}",
            )


# ── Helpers used by graders ───────────────────────────────────────────────────

def run_query(con: sqlite3.Connection, sql: str, iterations: int = 50) -> tuple[list, float]:
    """Execute sql, return (rows, avg_time_ms). Raises on syntax error."""
    cursor = con.cursor()
    # Warm up
    cursor.execute(sql)
    rows = [dict(r) for r in cursor.fetchall()]

    # Time it
    start = time.perf_counter()
    for _ in range(iterations):
        cursor.execute(sql)
        cursor.fetchall()
    elapsed_ms = (time.perf_counter() - start) / iterations * 1000
    return rows, elapsed_ms
