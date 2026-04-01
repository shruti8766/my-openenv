from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    NONE = "none"


class ActionDecision(str, Enum):
    IGNORE = "ignore"
    ALERT_ADMIN = "alert_admin"
    RESTART_SERVICE = "restart_service"
    ESCALATE = "escalate"


class LogEntry(BaseModel):
    timestamp: str
    service: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    source_ip: Optional[str] = None
    user: Optional[str] = None
    is_anomalous: bool = False
    anomaly_type: Optional[str] = None


class SystemStatus(BaseModel):
    cpu_usage: float = Field(ge=0, le=100)
    memory_usage: float = Field(ge=0, le=100)
    disk_usage: float = Field(ge=0, le=100)
    network_in_mbps: float = Field(ge=0)
    network_out_mbps: float = Field(ge=0)
    services: Dict[str, str] = Field(default_factory=dict)  # service -> status
    active_connections: int = Field(ge=0)
    failed_login_attempts: int = Field(ge=0)


class Observation(BaseModel):
    logs: List[LogEntry]
    system_status: SystemStatus
    previous_actions: List[str] = Field(default_factory=list)
    step_count: int = 0
    task_id: str = "easy"


class Action(BaseModel):
    decision: ActionDecision
    severity_assessment: SeverityLevel = SeverityLevel.NONE
    anomaly_detected: bool = False
    anomalous_log_indices: List[int] = Field(default_factory=list)
    reasoning: Optional[str] = None


class Reward(BaseModel):
    total: float
    anomaly_detection_score: float = 0.0
    severity_classification_score: float = 0.0
    action_score: float = 0.0
    penalty: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
