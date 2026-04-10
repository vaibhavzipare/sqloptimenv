"""
app.py — FastAPI server exposing the OpenEnv HTTP interface for SQLOptimEnv
Endpoints: POST /reset  POST /step  GET /state  GET /health  GET /
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import Action, Observation, Reward, SQLOptimEnv, State
from tasks import TASKS

app = FastAPI(
    title="SQLOptimEnv",
    description="OpenEnv environment for SQL query optimization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared env instance (stateless per request via reset)
_env: Optional[SQLOptimEnv] = None


def get_env() -> SQLOptimEnv:
    global _env
    if _env is None:
        _env = SQLOptimEnv()
    return _env


# ── Request / Response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy_syntax_fix"


class StepRequest(BaseModel):
    rewritten_query: str


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "tasks": list(TASKS.keys())}


@app.get("/")
def root():
    return {
        "name": "SQLOptimEnv",
        "version": "1.0.0",
        "tasks": list(TASKS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    if req.task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Valid: {list(TASKS.keys())}",
        )
    env = get_env()
    obs = env.reset(task_id=req.task_id)
    return obs


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = get_env()
    try:
        action = Action(rewritten_query=req.rewritten_query)
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=State)
def state():
    env = get_env()
    return env.state()
