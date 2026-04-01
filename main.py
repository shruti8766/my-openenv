"""
SOC Environment API Server
Exposes /reset and /step endpoints compliant with OpenEnv spec.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import json

from env.environment import SOCEnvironment
from env.models import Action, ActionDecision, SeverityLevel
from grader.grader import grade_action, grade_task

app = FastAPI(
    title="SOC RL Environment",
    description="Security Operations Center Log Monitoring RL Environment",
    version="1.0.0",
)

# In-memory session store (single session for simplicity)
_sessions: Dict[str, SOCEnvironment] = {}


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None
    session_id: str = "default"


class StepRequest(BaseModel):
    session_id: str = "default"
    decision: str = "ignore"
    severity_assessment: str = "none"
    anomaly_detected: bool = False
    anomalous_log_indices: list = []
    reasoning: Optional[str] = None


@app.get("/")
def root():
    return {
        "status": "ok",
        "environment": "SOC Log Monitoring RL Environment",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(req: ResetRequest):
    """Initialize a new episode."""
    try:
        env = SOCEnvironment(task_id=req.task_id, seed=req.seed)
        obs = env.reset()
        _sessions[req.session_id] = env
        return {
            "observation": obs.model_dump(),
            "ground_truth_keys": list(env.ground_truth().keys()),
            "session_id": req.session_id,
            "task_id": req.task_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Take an action in the environment."""
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found. Call /reset first.")

    try:
        action = Action(
            decision=ActionDecision(req.decision),
            severity_assessment=SeverityLevel(req.severity_assessment),
            anomaly_detected=req.anomaly_detected,
            anomalous_log_indices=req.anomalous_log_indices,
            reasoning=req.reasoning,
        )
        result = env.step(action)
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward.model_dump(),
            "done": result.done,
            "info": result.info,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state(session_id: str = "default"):
    """Get current environment state."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    obs = env.state()
    if obs is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return {"observation": obs.model_dump()}


@app.post("/grade")
def grade(task_id: str, action_dict: Dict[str, Any], session_id: str = "default"):
    """Grade an action without stepping the environment."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    gt = env.ground_truth()
    score = grade_task(task_id, action_dict, gt)
    return {"score": score, "ground_truth": gt}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
