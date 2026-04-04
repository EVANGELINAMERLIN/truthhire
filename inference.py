import os
import json
import requests
from openai import OpenAI

# ─── Environment Variables (EXACT format required) ──────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")  # NO default!
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "https://merlin018-truthhire.hf.space")

# ─── OpenAI Client (required) ───────────────────────────────
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

# ─── Environment API calls ──────────────────────────────────
def reset(task_id="easy"):
    res = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id}
    )
    return res.json()

def step(action: dict):
    res = requests.post(
        f"{ENV_URL}/step",
        json=action
    )
    return res.json()

# ─── Run Single Task ────────────────────────────────────────
def run_task(task_id: str):
    obs = reset(task_id)
    done = False
    step_num = 0
    final_score = 0.0

    # ─── EXACT log format required ──────────────────────────
    print(json.dumps({
        "event": "START",
        "task_id": task_id,
        "document_type": obs.get("document_type"),
        "instructions": obs.get("instructions")
    }))

    while not done:
        # Ask AI to analyze document
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at detecting:
1. Biased language in job descriptions and medical research
2. AI-generated content in news articles

Always respond in this exact JSON format only, no other text:
{
    "bias_phrases": ["phrase1", "phrase2"],
    "ai_sentences": ["sentence1", "sentence2"],
    "severity": "low or medium or high",
    "explanation": "your explanation here"
}"""
                },
                {
                    "role": "user",
                    "content": f"""
Document Type: {obs.get('document_type')}
Instructions: {obs.get('instructions')}

Document:
{obs.get('document')}

Respond in JSON format only.
"""
                }
            ],
            max_tokens=500
        )

        # Parse AI response
        try:
            raw = response.choices[0].message.content
            clean = raw.replace("```json", "").replace("```", "").strip()
            action = json.loads(clean)
        except Exception:
            action = {
                "bias_phrases": [],
                "ai_sentences": [],
                "severity": "low",
                "explanation": "Could not parse response"
            }

        # ─── EXACT STEP log format ───────────────────────────
        print(json.dumps({
            "event": "STEP",
            "step": step_num,
            "action": action
        }))

        # Send action to environment
        result = step(action)
        obs = result.get("observation", obs)
        done = result.get("done", True)
        final_score = result.get("reward", {}).get("score", 0.0)
        step_num += 1

    # ─── EXACT END log format ────────────────────────────────
    print(json.dumps({
        "event": "END",
        "task_id": task_id,
        "score": final_score
    }))

    return final_score

# ─── Run All Tasks ───────────────────────────────────────────
if __name__ == "__main__":
    all_scores = {}

    for task in ["easy", "medium", "hard", "expert"]:
        score = run_task(task)
        all_scores[task] = score

    print(json.dumps({
        "event": "SUMMARY",
        "scores": all_scores,
        "average": round(sum(all_scores.values()) / len(all_scores), 2)
    }))
