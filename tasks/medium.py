"""
MEDIUM TASK: Multi-Log Anomaly Identification
- Input: 4-8 log entries + system status
- Goal: Identify which logs are anomalous, assess overall severity, choose action
- Score: Partial credit for partial identification
"""
from env.environment import SOCEnvironment
from env.models import Action, SeverityLevel, ActionDecision
from typing import Dict, Any


TASK_CONFIG = {
    "id": "medium",
    "name": "Multi-Log Anomaly Identification",
    "description": (
        "Given multiple system log entries and current system metrics, "
        "identify which log entries are anomalous, classify the overall "
        "severity, and recommend an appropriate response action."
    ),
    "objective": "Identify anomalous logs from a batch of log entries",
    "max_steps": 3,
    "scoring": {
        "anomaly_detection": 0.4,
        "severity_classification": 0.3,
        "action_decision": 0.3,
    },
}


def run_task(seed: int = 42) -> Dict[str, Any]:
    """Run the medium task and return results."""
    env = SOCEnvironment(task_id="medium", seed=seed)
    obs = env.reset()

    log_lines = []
    for i, log in enumerate(obs.logs):
        log_lines.append(
            f"[{i}] [{log.timestamp}] [{log.level}] {log.service}: {log.message} "
            f"| IP: {log.source_ip} | User: {log.user}"
        )
    logs_str = "\n".join(log_lines)

    svc_str = ", ".join(f"{k}={v}" for k, v in obs.system_status.services.items())
    prompt = f"""You are a Security Operations Center analyst reviewing a batch of logs.

SYSTEM STATUS:
- CPU: {obs.system_status.cpu_usage}% | Memory: {obs.system_status.memory_usage}% | Disk: {obs.system_status.disk_usage}%
- Network In: {obs.system_status.network_in_mbps} Mbps | Out: {obs.system_status.network_out_mbps} Mbps
- Active Connections: {obs.system_status.active_connections}
- Failed Login Attempts: {obs.system_status.failed_login_attempts}
- Services: {svc_str}

LOG ENTRIES (index: timestamp, level, service: message):
{logs_str}

TASK: Analyze these logs and respond with a JSON object:
{{
  "anomaly_detected": true/false,
  "anomalous_log_indices": [list of 0-based indices of anomalous logs],
  "severity_assessment": "none" | "low" | "medium" | "high",
  "decision": "ignore" | "alert_admin" | "restart_service" | "escalate",
  "reasoning": "Brief explanation of which logs triggered concern and why"
}}
"""
    return {
        "task_id": "medium",
        "observation": obs,
        "prompt": prompt,
        "env": env,
        "ground_truth": env.ground_truth(),
    }


def score_response(env: SOCEnvironment, action: Action) -> float:
    result = env.step(action)
    return result.reward.total
