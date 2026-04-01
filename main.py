"""
SOC Environment API Server
Exposes /reset and /step endpoints compliant with OpenEnv spec.
"""
from fastapi import FastAPI, HTTPException, Request
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

# In-memory session store
_sessions: Dict[str, SOCEnvironment] = {}


class StepRequest(BaseModel):
    session_id: Optional[str] = "default"
    decision: Optional[str] = "ignore"
    severity_assessment: Optional[str] = "none"
    anomaly_detected: Optional[bool] = False
    anomalous_log_indices: Optional[list] = []
    reasoning: Optional[str] = None

    model_config = {"extra": "ignore"}


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
async def reset(request: Request):
    """Initialize a new episode. Accepts empty, null, or JSON body."""
    # Parse body safely — works with empty, null, or missing body
    try:
        body = await request.body()
        if body:
            data = json.loads(body)
            if not isinstance(data, dict):
                data = {}
        else:
            data = {}
    except Exception:
        data = {}

    task_id = data.get("task_id", "easy")
    seed = data.get("seed", None)
    session_id = data.get("session_id", "default")

    try:
        env = SOCEnvironment(task_id=task_id, seed=seed)
        obs = env.reset()
        _sessions[session_id] = env
        return {
            "observation": obs.model_dump(),
            "ground_truth_keys": list(env.ground_truth().keys()),
            "session_id": session_id,
            "task_id": task_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(request: Request):
    """Take an action. Accepts empty or JSON body."""
    try:
        body = await request.body()
        if body:
            data = json.loads(body)
            if not isinstance(data, dict):
                data = {}
        else:
            data = {}
    except Exception:
        data = {}

    session_id = data.get("session_id", "default")
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")

    try:
        action = Action(
            decision=ActionDecision(data.get("decision", "ignore")),
            severity_assessment=SeverityLevel(data.get("severity_assessment", "none")),
            anomaly_detected=bool(data.get("anomaly_detected", False)),
            anomalous_log_indices=data.get("anomalous_log_indices", []),
            reasoning=data.get("reasoning", None),
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