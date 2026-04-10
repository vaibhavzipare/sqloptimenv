"""
app.py — FastAPI server exposing the OpenEnv HTTP interface for SQLOptimEnv
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import Action, Observation, Reward, SQLOptimEnv, State
from tasks import TASKS

app = FastAPI(title="SQLOptimEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[SQLOptimEnv] = None

def get_env() -> SQLOptimEnv:
    global _env
    if _env is None:
        _env = SQLOptimEnv()
    return _env

class StepRequest(BaseModel):
    rewritten_query: str

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

@app.get("/health")
def health():
    return {"status": "ok", "tasks": list(TASKS.keys())}

@app.get("/")
def root():
    return {"name": "SQLOptimEnv", "version": "1.0.0", "tasks": list(TASKS.keys())}

@app.post("/reset", response_model=Observation)
async def reset(request: Request):
    try:
        body = await request.body()
        import json
        data = json.loads(body) if body else {}
    except Exception:
        data = {}
    task_id = data.get("task_id", "easy_syntax_fix") if data else "easy_syntax_fix"
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")
    return get_env().reset(task_id=task_id)

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    try:
        action = Action(rewritten_query=req.rewritten_query)
        obs, reward, done, info = get_env().step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=State)
def state():
    return get_env().state()