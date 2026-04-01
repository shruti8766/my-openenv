---
title: SOC OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
# 🛡️ SOC Log Monitoring RL Environment

A production-quality reinforcement learning environment simulating a **Security Operations Center (SOC)** log monitoring and incident response workflow.

---

## 🎯 Problem Description

Security Operations Centers generate thousands of log entries per minute across dozens of services. Human analysts must:
1. Detect anomalies hidden within normal traffic
2. Classify the severity of detected threats
3. Decide on the appropriate response action

This environment challenges an AI agent to perform all three steps, mirroring the real decision-making process of a SOC analyst.

---

## 🌍 Real-World Motivation

SOC teams face:
- **Alert fatigue**: Too many false positives cause real threats to be missed
- **Response latency**: Manual triage takes 10–60 minutes; automation can cut this to seconds
- **Skill gap**: Experienced analysts are scarce; AI can augment junior analysts

This environment trains models that could assist or automate Tier-1 SOC triage.

---

## 📊 Observation Space

```python
Observation(
    logs=[
        LogEntry(
            timestamp="2024-06-01T12:00:00Z",
            service="auth",
            level="ERROR",            # INFO | WARNING | ERROR | CRITICAL
            message="Failed login attempt 23 for user admin from 203.0.113.42",
            source_ip="203.0.113.42",
            user="admin",
        )
    ],
    system_status=SystemStatus(
        cpu_usage=87.3,               # % [0, 100]
        memory_usage=91.2,            # % [0, 100]
        disk_usage=65.0,              # % [0, 100]
        network_in_mbps=450.0,
        network_out_mbps=120.0,
        services={"auth": "running", "database": "degraded"},
        active_connections=1850,
        failed_login_attempts=47,
    ),
    previous_actions=["alert_admin"],
    step_count=1,
    task_id="hard",
)
```

---

## ⚡ Action Space

```python
Action(
    decision="escalate",              # ignore | alert_admin | restart_service | escalate
    severity_assessment="high",       # none | low | medium | high
    anomaly_detected=True,
    anomalous_log_indices=[2, 5, 7],  # which logs are anomalous
    reasoning="Brute force attack detected from 203.0.113.42 ...",
)
```

---

## 🏆 Reward Design

| Component | Max Score | Description |
|-----------|-----------|-------------|
| Anomaly Detection | +0.40 | Correct yes/no + F1 over identified indices |
| Severity Classification | +0.30 | Exact match; +0.15 for off-by-one |
| Action Decision | +0.30 | Correct action; +0.15 for acceptable alternative |

**Penalties:**
- Missed anomaly (false negative): −0.20
- Ignored critical incident: −0.15
- Wrong action with no partial credit: −0.15

**Total reward range:** [−0.35, 1.0], clipped to [0.0, 1.0]

---

## 📋 Task Descriptions

### 🟢 EASY — Single Log Anomaly Detection
- **Input:** 1 log entry + system status
- **Goal:** Detect if anomalous; if so, classify severity + choose action
- **Max Steps:** 1
- **Baseline Score:** ~0.65

### 🟡 MEDIUM — Multi-Log Anomaly Identification
- **Input:** 4–8 log entries + system status
- **Goal:** Identify which specific logs are anomalous; assess overall severity
- **Max Steps:** 3
- **Baseline Score:** ~0.50

### 🔴 HARD — Full SOC Incident Response
- **Input:** 6–12 logs showing coordinated attack + degrading system metrics
- **Goal:** Detect incident pattern, pinpoint all anomalous logs, classify severity, recommend action
- **Max Steps:** 5
- **Baseline Score:** ~0.40

---

## 🚀 Setup Instructions

### Local Development

```bash
# Clone and install
git clone <repo-url>
cd soc_env
pip install -r requirements.txt

# Start the API server
python main.py
# Server runs at http://localhost:7860
```

### Docker

```bash
# Build
docker build -t soc-env .

# Run
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_token \
  soc-env
```

### Running Inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_openai_key

python inference.py
```

---

## 🔌 API Usage

### Reset Environment
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard", "seed": 42, "session_id": "my_session"}'
```

### Take Action
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my_session",
    "decision": "escalate",
    "severity_assessment": "high",
    "anomaly_detected": true,
    "anomalous_log_indices": [2, 5],
    "reasoning": "Brute force detected"
  }'
```

### Response
```json
{
  "observation": { "logs": [...], "system_status": {...}, ... },
  "reward": {
    "total": 0.85,
    "anomaly_detection_score": 0.35,
    "severity_classification_score": 0.3,
    "action_score": 0.3,
    "penalty": 0.1
  },
  "done": false,
  "info": { "ground_truth": {...}, "step": 1, "task_id": "hard" }
}
```

---

## 📁 Project Structure

```
soc_env/
├── env/
│   ├── environment.py     # Core SOCEnvironment class
│   └── models.py          # Pydantic models (Observation, Action, Reward)
├── tasks/
│   ├── easy.py            # Single log detection
│   ├── medium.py          # Multi-log identification
│   └── hard.py            # Full incident response
├── grader/
│   └── grader.py          # Deterministic scoring functions
├── main.py                # FastAPI server
├── inference.py           # LLM baseline runner
├── openenv.yaml           # OpenEnv spec compliance
├── Dockerfile             # Container for HF Spaces
├── requirements.txt
└── README.md
```

---

## 📈 Baseline Scores

| Task | Random Agent | GPT-4o-mini | GPT-4o |
|------|-------------|-------------|--------|
| Easy | 0.15 | ~0.65 | ~0.82 |
| Medium | 0.10 | ~0.50 | ~0.70 |
| Hard | 0.08 | ~0.40 | ~0.62 |
| **Overall** | **0.11** | **~0.52** | **~0.71** |

---

## 🔁 Reproducibility

All runs use seeded random generation. Same seed → same environment state → same score.

```python
env = SOCEnvironment(task_id="hard", seed=42)
obs = env.reset()  # Always identical for seed=42
```

---
## ⚙️ Environment Variables
 
| Variable | Description |
|---|---|
| `API_BASE_URL` | OpenAI-compatible API base URL |
| `MODEL_NAME` | Model to use (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | HuggingFace / API token |
 
---
 
## 🔐 Security — API Keys & Tokens
 
**Never hardcode your token in any file.** Always use environment variables.
 
### ✅ Correct way — set in terminal before running
 
```powershell
# Windows PowerShell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN = "hf_your_token_here"
 
python inference.py
```
 
```bash
# Linux / Mac
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
 
python inference.py
```
 
### ✅ For HuggingFace Space — use Secrets
 
Go to your Space → **Settings** → **Variables and Secrets** → add:
 
| Secret Name | Value |
|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | `hf_your_token_here` |
 
HuggingFace injects these automatically at runtime — they are never visible in your code or logs.
 
### ❌ Never do this
 
```python
# DON'T commit tokens in code
HF_TOKEN = "hf_abc123..."  # ← anyone can steal this from GitHub
```
 
### 📄 Recommended .gitignore
 
Create a `.gitignore` file in your project root:
 
```
.env
inference_results.json
__pycache__/
*.pyc
*.pyc
.sixth/
```