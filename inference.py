import os
import json
import requests
from openai import OpenAI

# ─── Environment Variables ───────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")  # NO default!
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "https://merlin018-truthhire.hf.space")

# ─── OpenAI Client ───────────────────────────────────────────
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

# ─── Environment API calls ──────────────────────────────────
def reset(task_id="easy"):
    try:
        res = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id},
            timeout=30
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(json.dumps({"event": "ERROR", "stage": "reset", "error": str(e)}), flush=True)
        return {}

def step(action: dict):
    try:
        res = requests.post(
            f"{ENV_URL}/step",
            json=action,
            timeout=30
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(json.dumps({"event": "ERROR", "stage": "step", "error": str(e)}), flush=True)
        return {"observation": {}, "reward": 0.0, "done": True, "info": {}}

# ─── Extract score safely ────────────────────────────────────
def extract_score(reward):
    if isinstance(reward, dict):
        score = reward.get("score", 0.1)
    elif isinstance(reward, (int, float)):
        score = float(reward)
    else:
        score = 0.1
    # Must be strictly between 0 and 1 (not 0.0, not 1.0)
    score = max(0.01, min(0.99, score))
    return score

# ─── Run Single Task ────────────────────────────────────────
def run_task(task_id: str):
    obs = reset(task_id)
    done = False
    step_num = 0
    final_score = 0.1  # default > 0

    # ─── Required [START] block ──────────────────────────────
    print(f"[START] task={task_id}", flush=True)

    while not done:
        # Ask AI to analyze document
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at detecting:
1. Biased language in job descriptions
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

Analyze the document carefully and respond in JSON format only.
Identify ALL biased phrases and AI-generated sentences you find.
Be thorough - missing items reduces your score.
"""
                    }
                ],
                max_tokens=500
            )
            raw = response.choices[0].message.content
            clean = raw.replace("```json", "").replace("```", "").strip()
            action = json.loads(clean)

        except json.JSONDecodeError:
            action = {
                "bias_phrases": [],
                "ai_sentences": [],
                "severity": "low",
                "explanation": "Could not parse response"
            }
        except Exception as e:
            action = {
                "bias_phrases": [],
                "ai_sentences": [],
                "severity": "low",
                "explanation": f"Error: {str(e)}"
            }

        # Send action to environment
        result = step(action)
        obs = result.get("observation", obs)
        done = result.get("done", True)
        reward = extract_score(result.get("reward", 0.1))
        final_score = reward
        step_num += 1

        # ─── Required [STEP] block ───────────────────────────
        print(f"[STEP] step={step_num} reward={reward}", flush=True)

    # ─── Required [END] block ────────────────────────────────
    print(f"[END] task={task_id} score={final_score} steps={step_num}", flush=True)

    return final_score

# ─── Run All Tasks ───────────────────────────────────────────
if __name__ == "__main__":
    all_scores = {}

    for task in ["easy", "medium", "hard"]:
        score = run_task(task)
        all_scores[task] = score

    print(json.dumps({
        "event": "SUMMARY",
        "scores": all_scores,
        "average": round(sum(all_scores.values()) / len(all_scores), 2)
    }), flush=True)