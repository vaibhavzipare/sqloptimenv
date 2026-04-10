"""
SQLOptimEnv — OpenEnv-compliant SQL Query Optimization Environment
"""

import sqlite3
import traceback
from typing import Any, Optional

from openenv.core import Action, Environment, Observation, State
from pydantic import Field

from sqloptimenv.tasks import TASKS


class SQLObservation(Observation):
    task_id: str = Field(default="easy_syntax_fix")
    description: str = Field(default="")
    schema_sql: str = Field(default="")
    original_query: str = Field(default="")
    hint: Optional[str] = Field(default=None)
    step: int = Field(default=0)
    last_error: Optional[str] = Field(default=None)


class SQLAction(Action):
    rewritten_query: str = Field(default="")


class SQLState(State):
    task_id: str = Field(default="easy_syntax_fix")
    is_done: bool = Field(default=False)
    last_reward: float = Field(default=0.0)
    last_error: Optional[str] = Field(default=None)


class SQLOptimEnv(Environment[SQLAction, SQLObservation, SQLState]):
    """OpenEnv environment for SQL query optimization."""

    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_STEPS = 5

    def __init__(self, task_id: str = "easy_syntax_fix", **kwargs):
        super().__init__(**kwargs)
        self._task_id = task_id
        self._step_count = 0
        self._is_done = False
        self._last_reward: float = 0.0
        self._last_error: Optional[str] = None
        self._con: Optional[sqlite3.Connection] = None
        self._task = TASKS[task_id]

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs) -> SQLObservation:
        if task_id and task_id in TASKS:
            self._task_id = task_id
            self._task = TASKS[task_id]
        self._step_count = 0
        self._is_done = False
        self._last_reward = 0.0
        self._last_error = None
        if self._con:
            self._con.close()
        self._con = sqlite3.connect(":memory:")
        self._con.row_factory = sqlite3.Row
        for stmt in self._task["setup_sql"]:
            self._con.executescript(stmt)
        self._con.commit()
        return self._build_obs()

    def step(self, action: SQLAction, timeout_s=None, **kwargs) -> SQLObservation:
        if self._is_done:
            raise RuntimeError("Episode done — call reset() first.")
        self._step_count += 1
        result = self._grade(action.rewritten_query)
        self._last_reward = result["score"]
        self._last_error = result.get("error")
        if result["correct"] or self._step_count >= self.MAX_STEPS:
            self._is_done = True
        obs = self._build_obs()
        obs.reward = result["score"]
        obs.done = self._is_done
        obs.metadata = result
        return obs

    @property
    def state(self) -> SQLState:
        return SQLState(
            task_id=self._task_id,
            step_count=self._step_count,
            is_done=self._is_done,
            last_reward=self._last_reward,
            last_error=self._last_error,
        )

    def _build_obs(self) -> SQLObservation:
        return SQLObservation(
            task_id=self._task_id,
            description=self._task["description"],
            schema_sql=self._task["schema_sql"],
            original_query=self._task["original_query"],
            hint=self._task.get("hint"),
            step=self._step_count,
            last_error=self._last_error,
            reward=self._last_reward,
            done=self._is_done,
        )

    def _grade(self, rewritten_query: str) -> dict:
        grader = self._task["grader"]
        try:
            return grader(self._con, self._task["original_query"], rewritten_query)
        except Exception:
            return {"score": 0.0, "correct": False, "faster": False,
                    "error": traceback.format_exc(limit=2), "speedup": 0.0}
