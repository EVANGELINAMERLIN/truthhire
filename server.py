from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from environment import TruthHireEnv, Observation, Action, Reward

app = FastAPI()
env = TruthHireEnv()

# ─── Request Models ────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class StepRequest(BaseModel):
    bias_phrases: list[str] = []
    ai_sentences: list[str] = []
    severity: str = "low"
    explanation: str = ""

# ─── Routes ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "TruthHire Environment is running!"}

@app.post("/reset")
def reset(request: ResetRequest):
    obs = env.reset(task_id=request.task_id)
    return obs

@app.post("/step")
def step(request: StepRequest):
    action = Action(
        bias_phrases=request.bias_phrases,
        ai_sentences=request.ai_sentences,
        severity=request.severity,
        explanation=request.explanation
    )
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": "easy", "description": "Detect obvious bias in job description"},
            {"id": "medium", "description": "Detect AI-generated sentences in news"},
            {"id": "hard", "description": "Detect both bias and AI content together"}
        ]
    }




