"""
EASY TASK: Single Log Anomaly Detection
- Input: One log entry + system status
- Goal: Detect whether the log is anomalous (yes/no)
- Score: Binary + partial from severity
"""
from env.environment import SOCEnvironment
from env.models import Action, SeverityLevel, ActionDecision
from grader.grader import grade_action
from typing import Dict, Any


TASK_CONFIG = {
    "id": "easy",
    "name": "Single Log Anomaly Detection",
    "description": (
        "Given a single system log entry and current system metrics, "
        "determine whether the log represents an anomaly. "
        "If anomalous, classify severity and choose an appropriate action."
    ),
    "objective": "Detect anomaly in a single log entry",
    "max_steps": 1,
    "scoring": {
        "anomaly_detection": 0.4,
        "severity_classification": 0.3,
        "action_decision": 0.3,
    },
}


def run_task(seed: int = 42) -> Dict[str, Any]:
    """Run the easy task and return results."""
    env = SOCEnvironment(task_id="easy", seed=seed)
    obs = env.reset()

    # Build prompt for LLM
    log = obs.logs[0]
    prompt = f"""You are a Security Operations Center analyst.

SYSTEM STATUS:
- CPU: {obs.system_status.cpu_usage}%
- Memory: {obs.system_status.memory_usage}%
- Failed login attempts: {obs.system_status.failed_login_attempts}
- Services: {obs.system_status.services}

LOG ENTRY:
[{log.timestamp}] [{log.level}] {log.service}: {log.message}
Source IP: {log.source_ip} | User: {log.user}

TASK: Analyze this single log entry and respond with a JSON object:
{{
  "anomaly_detected": true/false,
  "severity_assessment": "none" | "low" | "medium" | "high",
  "decision": "ignore" | "alert_admin" | "restart_service" | "escalate",
  "anomalous_log_indices": [0] if anomalous else [],
  "reasoning": "Brief explanation"
}}
"""
    return {
        "task_id": "easy",
        "observation": obs,
        "prompt": prompt,
        "env": env,
        "ground_truth": env.ground_truth(),
    }


def score_response(env: SOCEnvironment, action: Action) -> float:
    result = env.step(action)
    return result.reward.total
