import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from env.models import (
    LogEntry, SystemStatus, Observation, Action, Reward, StepResult,
    SeverityLevel, ActionDecision
)

SERVICES = ["auth", "api-gateway", "database", "web-server", "cache", "scheduler", "payment"]

NORMAL_MESSAGES = [
    ("INFO", "Request processed successfully in {ms}ms"),
    ("INFO", "User {user} logged in from {ip}"),
    ("INFO", "Health check passed for {service}"),
    ("INFO", "Scheduled job completed: backup"),
    ("INFO", "Cache hit ratio: {ratio}%"),
    ("INFO", "Database connection pool: {n}/{max} used"),
    ("INFO", "SSL certificate valid, expires in {days} days"),
]

ANOMALY_TEMPLATES = {
    "brute_force": {
        "level": "ERROR",
        "message": "Failed login attempt {n} for user {user} from {ip}",
        "severity": "high",
        "action": "escalate",
    },
    "ddos": {
        "level": "CRITICAL",
        "message": "Request rate limit exceeded: {rps} req/s from {ip}",
        "severity": "high",
        "action": "escalate",
    },
    "service_crash": {
        "level": "CRITICAL",
        "message": "Service {service} crashed unexpectedly. Exit code: 1",
        "severity": "high",
        "action": "restart_service",
    },
    "memory_leak": {
        "level": "WARNING",
        "message": "Memory usage spike detected: {pct}% on {service}",
        "severity": "medium",
        "action": "alert_admin",
    },
    "disk_full": {
        "level": "ERROR",
        "message": "Disk usage critical: {pct}% on /var/log",
        "severity": "medium",
        "action": "alert_admin",
    },
    "unauthorized_access": {
        "level": "ERROR",
        "message": "Unauthorized access attempt to /admin from {ip} by user {user}",
        "severity": "high",
        "action": "escalate",
    },
    "slow_query": {
        "level": "WARNING",
        "message": "Slow database query detected: {ms}ms for query on {table}",
        "severity": "low",
        "action": "ignore",
    },
    "config_change": {
        "level": "WARNING",
        "message": "Configuration file modified by {user}: /etc/{service}/config.yaml",
        "severity": "medium",
        "action": "alert_admin",
    },
}

SAMPLE_IPS = ["192.168.1.100", "10.0.0.55", "203.0.113.42", "198.51.100.7", "172.16.0.200"]
SAMPLE_USERS = ["admin", "root", "john.doe", "svc_account", "deploy_bot"]
SAMPLE_TABLES = ["users", "transactions", "sessions", "products", "audit_log"]


def _random_ip():
    return random.choice(SAMPLE_IPS)


def _random_user():
    return random.choice(SAMPLE_USERS)


def _ts(base: datetime, offset_secs: int = 0) -> str:
    return (base + timedelta(seconds=offset_secs)).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_normal_log(base_time: datetime, offset: int) -> LogEntry:
    template = random.choice(NORMAL_MESSAGES)
    level, msg_tpl = template
    message = msg_tpl.format(
        ms=random.randint(10, 300),
        user=_random_user(),
        ip=_random_ip(),
        service=random.choice(SERVICES),
        ratio=random.randint(70, 98),
        n=random.randint(1, 10),
        max=20,
        days=random.randint(30, 365),
        rps=random.randint(10, 100),
    )
    return LogEntry(
        timestamp=_ts(base_time, offset),
        service=random.choice(SERVICES),
        level=level,
        message=message,
        source_ip=_random_ip(),
        user=_random_user(),
        is_anomalous=False,
    )


def generate_anomaly_log(base_time: datetime, offset: int, anomaly_type: Optional[str] = None) -> LogEntry:
    if anomaly_type is None:
        anomaly_type = random.choice(list(ANOMALY_TEMPLATES.keys()))
    tmpl = ANOMALY_TEMPLATES[anomaly_type]
    message = tmpl["message"].format(
        n=random.randint(5, 50),
        user=_random_user(),
        ip=_random_ip(),
        service=random.choice(SERVICES),
        rps=random.randint(500, 5000),
        pct=random.randint(85, 99),
        ms=random.randint(5000, 30000),
        table=random.choice(SAMPLE_TABLES),
    )
    return LogEntry(
        timestamp=_ts(base_time, offset),
        service=random.choice(SERVICES),
        level=tmpl["level"],
        message=message,
        source_ip=_random_ip(),
        user=_random_user(),
        is_anomalous=True,
        anomaly_type=anomaly_type,
    )


def generate_system_status(stressed: bool = False) -> SystemStatus:
    if stressed:
        cpu = random.uniform(75, 99)
        memory = random.uniform(80, 99)
        disk = random.uniform(85, 99)
        failed_logins = random.randint(10, 100)
        connections = random.randint(500, 2000)
        svc_status = {s: random.choice(["running", "running", "degraded", "stopped"]) for s in SERVICES}
    else:
        cpu = random.uniform(10, 50)
        memory = random.uniform(20, 60)
        disk = random.uniform(30, 70)
        failed_logins = random.randint(0, 3)
        connections = random.randint(10, 200)
        svc_status = {s: "running" for s in SERVICES}
    return SystemStatus(
        cpu_usage=round(cpu, 1),
        memory_usage=round(memory, 1),
        disk_usage=round(disk, 1),
        network_in_mbps=round(random.uniform(1, 500), 1),
        network_out_mbps=round(random.uniform(1, 200), 1),
        services=svc_status,
        active_connections=connections,
        failed_login_attempts=failed_logins,
    )


def get_expected_action_for_severity(severity: str) -> str:
    mapping = {
        "high": "escalate",
        "medium": "alert_admin",
        "low": "ignore",
        "none": "ignore",
    }
    # service crash always wants restart
    return mapping.get(severity, "ignore")


class SOCEnvironment:
    def __init__(self, task_id: str = "easy", seed: Optional[int] = None):
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)
        self._state: Optional[Observation] = None
        self._ground_truth: Dict[str, Any] = {}
        self._step_count = 0
        self._max_steps = {"easy": 1, "medium": 3, "hard": 5}.get(task_id, 3)
        self._done = False

    def reset(self) -> Observation:
        self._step_count = 0
        self._done = False
        base_time = datetime(2024, 6, 1, 12, 0, 0)

        if self.task_id == "easy":
            has_anomaly = self._rng.choice([True, False])
            if has_anomaly:
                log = generate_anomaly_log(base_time, 0)
            else:
                log = generate_normal_log(base_time, 0)
            logs = [log]
            status = generate_system_status(stressed=has_anomaly)
            self._ground_truth = {
                "anomaly_detected": has_anomaly,
                "anomaly_indices": [0] if has_anomaly else [],
                "severity": ANOMALY_TEMPLATES[log.anomaly_type]["severity"] if has_anomaly else "none",
                "expected_action": ANOMALY_TEMPLATES[log.anomaly_type]["action"] if has_anomaly else "ignore",
            }

        elif self.task_id == "medium":
            n_logs = self._rng.randint(4, 8)
            logs = []
            anomaly_indices = []
            n_anomalies = self._rng.randint(1, 3)
            anomaly_positions = set(self._rng.sample(range(n_logs), min(n_anomalies, n_logs)))
            highest_severity = "none"
            expected_action = "ignore"
            for i in range(n_logs):
                if i in anomaly_positions:
                    log = generate_anomaly_log(base_time, i * 30)
                    anomaly_indices.append(i)
                    sev = ANOMALY_TEMPLATES[log.anomaly_type]["severity"]
                    if ["none", "low", "medium", "high"].index(sev) > ["none", "low", "medium", "high"].index(highest_severity):
                        highest_severity = sev
                        expected_action = ANOMALY_TEMPLATES[log.anomaly_type]["action"]
                else:
                    log = generate_normal_log(base_time, i * 30)
                logs.append(log)
            status = generate_system_status(stressed=(highest_severity == "high"))
            self._ground_truth = {
                "anomaly_detected": len(anomaly_indices) > 0,
                "anomaly_indices": anomaly_indices,
                "severity": highest_severity,
                "expected_action": expected_action,
            }

        else:  # hard
            n_logs = self._rng.randint(6, 12)
            logs = []
            anomaly_indices = []
            anomaly_type = self._rng.choice(list(ANOMALY_TEMPLATES.keys()))
            n_anomalies = self._rng.randint(2, 4)
            anomaly_positions = set(self._rng.sample(range(n_logs), min(n_anomalies, n_logs)))
            for i in range(n_logs):
                if i in anomaly_positions:
                    log = generate_anomaly_log(base_time, i * 15, anomaly_type=anomaly_type)
                    anomaly_indices.append(i)
                else:
                    log = generate_normal_log(base_time, i * 15)
                logs.append(log)
            severity = ANOMALY_TEMPLATES[anomaly_type]["severity"]
            stressed = severity in ("high", "medium")
            status = generate_system_status(stressed=stressed)
            self._ground_truth = {
                "anomaly_detected": True,
                "anomaly_indices": anomaly_indices,
                "anomaly_type": anomaly_type,
                "severity": severity,
                "expected_action": ANOMALY_TEMPLATES[anomaly_type]["action"],
            }

        self._state = Observation(
            logs=logs,
            system_status=status,
            previous_actions=[],
            step_count=0,
            task_id=self.task_id,
        )
        return self._state

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise ValueError("Episode is done. Call reset() first.")
        if self._state is None:
            raise ValueError("Call reset() before step().")

        reward = self._compute_reward(action)
        self._step_count += 1
        self._done = self._step_count >= self._max_steps

        # Update system state based on action
        new_status = self._apply_action_effects(action, self._state.system_status)
        new_obs = Observation(
            logs=self._state.logs,
            system_status=new_status,
            previous_actions=self._state.previous_actions + [action.decision.value],
            step_count=self._step_count,
            task_id=self.task_id,
        )
        self._state = new_obs

        return StepResult(
            observation=new_obs,
            reward=reward,
            done=self._done,
            info={
                "ground_truth": self._ground_truth,
                "step": self._step_count,
                "task_id": self.task_id,
            }
        )

    def _compute_reward(self, action: Action) -> Reward:
        gt = self._ground_truth
        anomaly_score = 0.0
        severity_score = 0.0
        action_score = 0.0
        penalty = 0.0

        # Anomaly detection component
        if action.anomaly_detected == gt["anomaly_detected"]:
            anomaly_score = 0.4
            if gt["anomaly_detected"] and action.anomalous_log_indices:
                # partial credit for partial index overlap
                gt_set = set(gt["anomaly_indices"])
                pred_set = set(action.anomalous_log_indices)
                if gt_set:
                    precision = len(gt_set & pred_set) / len(pred_set) if pred_set else 0
                    recall = len(gt_set & pred_set) / len(gt_set)
                    f1 = 2 * precision * recall / (precision + recall + 1e-9)
                    anomaly_score = 0.4 * f1
        else:
            # False negative (missed anomaly) is worse than false positive
            if gt["anomaly_detected"] and not action.anomaly_detected:
                penalty += 0.2
            else:
                penalty += 0.1

        # Severity classification
        if action.severity_assessment.value == gt["severity"]:
            severity_score = 0.3
        elif self._severity_close(action.severity_assessment.value, gt["severity"]):
            severity_score = 0.15  # partial credit

        # Action decision
        if action.decision.value == gt["expected_action"]:
            action_score = 0.3
        elif self._action_acceptable(action.decision.value, gt["expected_action"]):
            action_score = 0.15
        else:
            penalty += 0.15  # wrong action with no partial credit

        total = max(0.0, anomaly_score + severity_score + action_score - penalty)
        total = min(1.0, total)

        return Reward(
            total=round(total, 4),
            anomaly_detection_score=anomaly_score,
            severity_classification_score=severity_score,
            action_score=action_score,
            penalty=penalty,
            breakdown={
                "anomaly": anomaly_score,
                "severity": severity_score,
                "action": action_score,
                "penalty": -penalty,
            }
        )

    def _severity_close(self, pred: str, gt: str) -> bool:
        order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        return abs(order.get(pred, 0) - order.get(gt, 0)) == 1

    def _action_acceptable(self, pred: str, gt: str) -> bool:
        acceptable = {
            "escalate": {"alert_admin"},
            "alert_admin": {"escalate"},
            "restart_service": {"alert_admin"},
        }
        return pred in acceptable.get(gt, set())

    def _apply_action_effects(self, action: Action, status: SystemStatus) -> SystemStatus:
        services = dict(status.services)
        cpu = status.cpu_usage
        memory = status.memory_usage
        failed = status.failed_login_attempts

        if action.decision == ActionDecision.RESTART_SERVICE:
            # restart resolves service crashes but causes brief CPU spike
            for svc in services:
                if services[svc] == "stopped":
                    services[svc] = "running"
            cpu = min(100, cpu + 10)
        elif action.decision == ActionDecision.ESCALATE:
            # escalation brings in human responders - reduces failed logins
            failed = max(0, failed - 5)
        elif action.decision == ActionDecision.IGNORE and self._ground_truth.get("anomaly_detected"):
            # ignoring real anomaly degrades system
            cpu = min(100, cpu + 5)
            memory = min(100, memory + 5)
            failed = failed + 2

        return SystemStatus(
            cpu_usage=round(cpu, 1),
            memory_usage=round(memory, 1),
            disk_usage=status.disk_usage,
            network_in_mbps=status.network_in_mbps,
            network_out_mbps=status.network_out_mbps,
            services=services,
            active_connections=status.active_connections,
            failed_login_attempts=failed,
        )

    def state(self) -> Optional[Observation]:
        return self._state

    def ground_truth(self) -> Dict[str, Any]:
        return self._ground_truth
