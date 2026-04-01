"""
Deterministic grader for SOC Environment.
Same input → same score. Always returns float in [0.0, 1.0].
"""
from typing import Dict, Any, Optional
from env.models import Action, SeverityLevel, ActionDecision


SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


def grade_action(action: Action, ground_truth: Dict[str, Any], task_id: str = "easy") -> Dict[str, float]:
    """
    Grade an action against ground truth.
    Returns a dict with component scores and total.
    All values are floats in [0.0, 1.0].
    """
    anomaly_score = _grade_anomaly_detection(action, ground_truth)
    severity_score = _grade_severity(action, ground_truth)
    action_score = _grade_action_decision(action, ground_truth)
    penalty = _compute_penalty(action, ground_truth)

    raw_total = anomaly_score + severity_score + action_score - penalty
    total = max(0.0, min(1.0, raw_total))

    return {
        "total": round(total, 4),
        "anomaly_detection": round(anomaly_score, 4),
        "severity_classification": round(severity_score, 4),
        "action_decision": round(action_score, 4),
        "penalty": round(penalty, 4),
    }


def _grade_anomaly_detection(action: Action, gt: Dict[str, Any]) -> float:
    """
    Full credit (0.4) if anomaly_detected matches ground truth.
    For true positives with index prediction, uses F1 over indices.
    For false detection: partial penalty applied separately.
    """
    gt_detected = gt["anomaly_detected"]
    pred_detected = action.anomaly_detected

    if not pred_detected and not gt_detected:
        return 0.4  # Correct true negative

    if not pred_detected and gt_detected:
        return 0.0  # False negative - miss

    if pred_detected and not gt_detected:
        return 0.0  # False positive

    # True positive - check index overlap
    gt_indices = set(gt.get("anomaly_indices", []))
    pred_indices = set(action.anomalous_log_indices)

    if not gt_indices:
        return 0.4

    if not pred_indices:
        return 0.2  # Detected but didn't pinpoint - partial credit

    precision = len(gt_indices & pred_indices) / len(pred_indices)
    recall = len(gt_indices & pred_indices) / len(gt_indices)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return round(0.4 * f1, 4)


def _grade_severity(action: Action, gt: Dict[str, Any]) -> float:
    """
    Full credit (0.3) for exact match.
    Half credit (0.15) for off-by-one severity.
    Zero otherwise.
    """
    pred = action.severity_assessment.value
    expected = gt.get("severity", "none")

    if pred == expected:
        return 0.3

    pred_idx = SEVERITY_ORDER.get(pred, 0)
    exp_idx = SEVERITY_ORDER.get(expected, 0)

    if abs(pred_idx - exp_idx) == 1:
        return 0.15

    return 0.0


def _grade_action_decision(action: Action, gt: Dict[str, Any]) -> float:
    """
    Full credit (0.3) for exact match.
    Partial credit (0.15) for acceptable alternative.
    Zero otherwise.
    """
    pred = action.decision.value
    expected = gt.get("expected_action", "ignore")

    if pred == expected:
        return 0.3

    ACCEPTABLE_ALTERNATIVES = {
        "escalate": {"alert_admin"},
        "alert_admin": {"escalate", "restart_service"},
        "restart_service": {"alert_admin"},
        "ignore": set(),
    }

    if pred in ACCEPTABLE_ALTERNATIVES.get(expected, set()):
        return 0.15

    return 0.0


def _compute_penalty(action: Action, gt: Dict[str, Any]) -> float:
    """
    Penalties for dangerous mistakes.
    """
    penalty = 0.0
    gt_detected = gt["anomaly_detected"]
    pred_detected = action.anomaly_detected

    # Miss a real anomaly
    if gt_detected and not pred_detected:
        penalty += 0.2

    # Ignore when should escalate
    if gt.get("expected_action") == "escalate" and action.decision.value == "ignore":
        penalty += 0.15

    # Restart when should escalate (under-response)
    if gt.get("expected_action") == "escalate" and action.decision.value == "restart_service":
        penalty += 0.05

    return min(penalty, 0.35)


def grade_task(task_id: str, action_dict: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """
    Convenience function: parse action from dict and grade it.
    Returns total score as float in [0.0, 1.0].
    """
    try:
        action = Action(
            decision=ActionDecision(action_dict.get("decision", "ignore")),
            severity_assessment=SeverityLevel(action_dict.get("severity_assessment", "none")),
            anomaly_detected=bool(action_dict.get("anomaly_detected", False)),
            anomalous_log_indices=action_dict.get("anomalous_log_indices", []),
            reasoning=action_dict.get("reasoning", ""),
        )
        scores = grade_action(action, ground_truth, task_id)
        return scores["total"]
    except Exception as e:
        return 0.0
