"""
inference.py - Baseline inference script for SOC RL Environment
Uses OpenAI-compatible client to run all 3 tasks and report scores.

Environment variables:
  API_BASE_URL  - Base URL for the API (e.g., https://api.openai.com/v1)
  MODEL_NAME    - Model to use (e.g., gpt-4o-mini, claude-sonnet-4-20250514)
  HF_TOKEN      - HuggingFace token (used as API key if needed)
"""

import os
import json
import sys
import time
from typing import Dict, Any, Optional

from openai import OpenAI

from env.environment import SOCEnvironment
from env.models import Action, ActionDecision, SeverityLevel
from grader.grader import grade_action
from tasks import easy, medium, hard


# ── Config ──────────────────────────────────────────────────────────────────
# Line 26-28 — change to:
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # ← never hardcode token here
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "sk-placeholder",
)

SEEDS = [42, 123, 7]  # Run 3 seeds per task for reproducibility


def call_llm(prompt: str, max_tokens: int = 512) -> str:
    """Call the LLM and return the text response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Security Operations Center (SOC) analyst. "
                    "Always respond with valid JSON only. No markdown, no explanation outside the JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,  # Deterministic
    )
    return response.choices[0].message.content.strip()


def parse_action(response_text: str) -> Dict[str, Any]:
    """Parse LLM JSON response into action dict."""
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return json.loads(text)


def run_task(task_module, seed: int) -> Dict[str, Any]:
    """Run a single task with a given seed and return score details."""
    task_data = task_module.run_task(seed=seed)
    env: SOCEnvironment = task_data["env"]
    prompt: str = task_data["prompt"]
    gt = task_data["ground_truth"]
    task_id = task_data["task_id"]

    try:
        response_text = call_llm(prompt)
        action_dict = parse_action(response_text)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse error: {e}")
        action_dict = {
            "decision": "ignore",
            "severity_assessment": "none",
            "anomaly_detected": False,
            "anomalous_log_indices": [],
        }
    except Exception as e:
        print(f"  ⚠️  LLM error: {e}")
        action_dict = {
            "decision": "ignore",
            "severity_assessment": "none",
            "anomaly_detected": False,
            "anomalous_log_indices": [],
        }

    # Build Action object
    try:
        action = Action(
            decision=ActionDecision(action_dict.get("decision", "ignore")),
            severity_assessment=SeverityLevel(action_dict.get("severity_assessment", "none")),
            anomaly_detected=bool(action_dict.get("anomaly_detected", False)),
            anomalous_log_indices=action_dict.get("anomalous_log_indices", []),
            reasoning=action_dict.get("reasoning", ""),
        )
    except Exception as e:
        print(f"  ⚠️  Action parse error: {e}")
        action = Action(decision=ActionDecision.IGNORE)

    scores = grade_action(action, gt, task_id)

    return {
        "task_id": task_id,
        "seed": seed,
        "score": scores["total"],
        "breakdown": {
            "anomaly_detection": scores["anomaly_detection"],
            "severity_classification": scores["severity_classification"],
            "action_decision": scores["action_decision"],
            "penalty": scores["penalty"],
        },
        "ground_truth": {k: v for k, v in gt.items() if k != "anomaly_indices"},
        "predicted": {
            "anomaly_detected": action.anomaly_detected,
            "severity": action.severity_assessment.value,
            "decision": action.decision.value,
        },
        "llm_reasoning": action_dict.get("reasoning", ""),
    }


def main():
    print("=" * 60)
    print("SOC RL Environment - Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print("=" * 60)

    task_modules = [
        ("EASY", easy),
        ("MEDIUM", medium),
        ("HARD", hard),
    ]

    all_results = {}
    summary_scores = {}

    for task_name, task_mod in task_modules:
        print(f"\n{'─'*40}")
        print(f"Running {task_name} task...")
        print(f"{'─'*40}")

        task_scores = []
        for seed in SEEDS:
            print(f"  Seed {seed}: ", end="", flush=True)
            result = run_task(task_mod, seed)
            task_scores.append(result["score"])

            print(f"Score={result['score']:.4f} | "
                  f"Anomaly={result['breakdown']['anomaly_detection']:.2f} | "
                  f"Severity={result['breakdown']['severity_classification']:.2f} | "
                  f"Action={result['breakdown']['action_decision']:.2f}")
            print(f"          GT: detected={result['ground_truth'].get('anomaly_detected')} "
                  f"severity={result['ground_truth'].get('severity')} "
                  f"action={result['ground_truth'].get('expected_action')}")
            print(f"          Pred: detected={result['predicted']['anomaly_detected']} "
                  f"severity={result['predicted']['severity']} "
                  f"action={result['predicted']['decision']}")

            all_results[f"{task_name.lower()}_seed{seed}"] = result
            time.sleep(0.5)  # Rate limiting courtesy

        avg = sum(task_scores) / len(task_scores)
        summary_scores[task_name] = avg
        print(f"\n  ✅ {task_name} Average Score: {avg:.4f}")

    # Overall summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for task_name, score in summary_scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_name:<8}: [{bar}] {score:.4f}")

    overall = sum(summary_scores.values()) / len(summary_scores)
    print(f"\n  OVERALL:  {overall:.4f}")
    print("=" * 60)

    # Write results to file for reproducibility
    with open("inference_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "api_base": API_BASE_URL,
            "seeds": SEEDS,
            "summary": summary_scores,
            "overall": overall,
            "details": all_results,
        }, f, indent=2, default=str)
    print("\nResults saved to inference_results.json")

    return overall


if __name__ == "__main__":
    score = main()
    sys.exit(0 if score >= 0.0 else 1)
