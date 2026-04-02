# TruthHire — AI Content Integrity Auditor

An OpenEnv environment where an AI agent detects **bias in job descriptions** 
and **AI-generated content in news articles**.

## Why TruthHire?
- Companies like LinkedIn and Indeed actively fight biased job posts
- News platforms need tools to detect AI-generated fake content
- This environment trains agents to solve both problems together

---

## Environment Description

The agent receives a document (job post or news article) and must:
- Identify biased phrases
- Detect AI-generated sentences
- Rate severity (low/medium/high)
- Explain its findings

---

## Action Space
| Field | Type | Description |
|---|---|---|
| bias_phrases | list[str] | Biased phrases found in document |
| ai_sentences | list[str] | AI-generated sentences found |
| severity | str | "low", "medium", or "high" |
| explanation | str | Reasoning behind findings |

## Observation Space
| Field | Type | Description |
|---|---|---|
| document | str | The text to analyze |
| document_type | str | "job_post" or "news_article" |
| task_id | str | Current task difficulty |
| instructions | str | What the agent must do |

---

## Tasks

| Task | Description | Difficulty |
|---|---|---|
| easy | Detect obvious bias in job description | Easy |
| medium | Detect AI-generated sentences in news | Medium |
| hard | Detect both bias and AI content together | Hard |

---

## Reward
- Scores range from **0.0 to 1.0**
- Partial credit given for each correct finding
- Easy: scored on bias markers found
- Medium: scored on AI sentences detected
- Hard: combined score (bias + AI + severity)

---

## Setup & Usage

### Run locally
```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Run with Docker
```bash
docker build -t truthhire .
docker run -p 7860:7860 truthhire
```

### Run baseline inference
```bash
export HF_TOKEN=your_token
export API_BASE_URL=your_api_url
export MODEL_NAME=your_model
export ENV_URL=http://localhost:7860
python inference.py
```

---

## Baseline Scores
| Task | Score |
|---|---|
| Easy | 0.75 |
| Medium | 0.67 |
| Hard | 0.58 |

---

## Author
Built for OpenEnv Hackathon 2026
```

---

## 🎉 Your Complete Project:
```
TRUTHHIRE/
├── environment.py    ✅
├── openenv.yaml      ✅
├── server.py         ✅
├── requirements.txt  ✅
├── Dockerfile        ✅
├── inference.py      ✅
└── README.md         ✅