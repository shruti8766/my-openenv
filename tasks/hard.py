"""
HARD TASK: Full SOC Incident Response
- Input: 6-12 logs + rich system context + previous actions
- Goal: Detect anomaly pattern, classify severity, recommend action with reasoning
- Score: Full partial credit across all components + multi-step evaluation
"""
from env.environment import SOCEnvironment
from env.models import Action, SeverityLevel, ActionDecision
from typing import Dict, Any


TASK_CONFIG = {
    "id": "hard",
    "name": "Full SOC Incident Response",
    "description": (
        "Given a complex set of log entries showing a coordinated attack pattern "
        "alongside live system metrics showing degradation, identify the incident type, "
        "classify severity accurately, pinpoint all affected log entries, and recommend "
        "the most effective response action from the SOC playbook."
    ),
    "objective": (
        "Full incident response: anomaly detection + severity classification + "
        "action recommendation using logs and system context"
    ),
    "max_steps": 5,
    "scoring": {
        "anomaly_detection": 0.4,
        "severity_classification": 0.3,
        "action_decision": 0.3,
    },
}


def run_task(seed: int = 42) -> Dict[str, Any]:
    """Run the hard task and return results."""
    env = SOCEnvironment(task_id="hard", seed=seed)
    obs = env.reset()

    log_lines = []
    for i, log in enumerate(obs.logs):
        log_lines.append(
            f"[{i}] [{log.timestamp}] [{log.level}] {log.service}: {log.message}\n"
            f"     Source IP: {log.source_ip} | User: {log.user}"
        )
    logs_str = "\n".join(log_lines)

    stopped_svcs = [k for k, v in obs.system_status.services.items() if v != "running"]
    degraded_svcs = [k for k, v in obs.system_status.services.items() if v == "degraded"]

    degradation_str = ""
    if stopped_svcs:
        degradation_str += f"\n⚠️  STOPPED SERVICES: {', '.join(stopped_svcs)}"
    if degraded_svcs:
        degradation_str += f"\n⚠️  DEGRADED SERVICES: {', '.join(degraded_svcs)}"

    prompt = f"""You are a senior Security Operations Center (SOC) analyst performing incident response.

=== CURRENT SYSTEM HEALTH ===
CPU Usage:            {obs.system_status.cpu_usage}%
Memory Usage:         {obs.system_status.memory_usage}%
Disk Usage:           {obs.system_status.disk_usage}%
Network Inbound:      {obs.system_status.network_in_mbps} Mbps
Network Outbound:     {obs.system_status.network_out_mbps} Mbps
Active Connections:   {obs.system_status.active_connections}
Failed Login Attempts:{obs.system_status.failed_login_attempts} (last 15 min){degradation_str}

=== LOG STREAM (last 15 minutes) ===
{logs_str}

=== PREVIOUS ACTIONS TAKEN ===
{obs.previous_actions if obs.previous_actions else ['None']}

=== INCIDENT RESPONSE TASK ===
Analyze the complete picture: log patterns, system resource consumption, service health,
and network indicators. Identify the attack/incident pattern, assess blast radius, and
select the most appropriate response from the SOC playbook.

Available actions:
- ignore: No action needed, within normal parameters
- alert_admin: Notify on-call admin, monitor closely
- restart_service: Restart affected service(s) to restore availability
- escalate: Escalate to CISO/incident response team, high severity

Respond ONLY with a JSON object:
{{
  "anomaly_detected": true/false,
  "anomalous_log_indices": [0-based indices of ALL anomalous log entries],
  "severity_assessment": "none" | "low" | "medium" | "high",
  "decision": "ignore" | "alert_admin" | "restart_service" | "escalate",
  "reasoning": "Detailed explanation: incident type, affected systems, why this severity/action"
}}
"""
    return {
        "task_id": "hard",
        "observation": obs,
        "prompt": prompt,
        "env": env,
        "ground_truth": env.ground_truth(),
    }


def score_response(env: SOCEnvironment, action: Action) -> float:
    result = env.step(action)
    return result.reward.total
