import os
import json
from openai import OpenAI

# ─── Setup ─────────────────────────────────────────────────

client = OpenAI(
    api_key=os.environ.get("HF_TOKEN"),
    base_url=os.environ.get("API_BASE_URL")
)
MODEL = os.environ.get("MODEL_NAME")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

import requests

def reset(task_id="easy"):
    res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    return res.json()

def step(action: dict):
    res = requests.post(f"{ENV_URL}/step", json=action)
    return res.json()

def get_state():
    res = requests.get(f"{ENV_URL}/state")
    return res.json()

# ─── AI Agent ──────────────────────────────────────────────

def run_task(task_id: str):
    print("[START]")
    print(f"task_id={task_id}")

    obs = reset(task_id)
    done = False
    step_count = 0
    final_score = 0.0

    while not done:
        # Ask AI to analyze the document
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at detecting:
                    1. Biased language in job descriptions
                    2. AI-generated content in news articles
                    
                    Always respond in this exact JSON format:
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
                    
                    Analyze and respond in JSON format only.
                    """
                }
            ]
        )

        # Parse AI response
        try:
            raw = response.choices[0].message.content
            clean = raw.replace("```json", "").replace("```", "").strip()
            action = json.loads(clean)
        except Exception as e:
            action = {
                "bias_phrases": [],
                "ai_sentences": [],
                "severity": "low",
                "explanation": "Could not parse response"
            }

        print(f"[STEP] step={step_count} action={json.dumps(action)}")

        # Send action to environment
        result = step(action)
        obs = result.get("observation", obs)
        done = result.get("done", True)
        final_score = result.get("reward", {}).get("score", 0.0)
        step_count += 1

    print(f"[END] task_id={task_id} score={final_score}")
    return final_score

# ─── Run All Tasks ─────────────────────────────────────────

if __name__ == "__main__":
    scores = {}
    for task in ["easy", "medium", "hard"]:
        score = run_task(task)
        scores[task] = score

    print("\n── Final Scores ──")
    for task, score in scores.items():
        print(f"{task}: {score}")
    print(f"Average: {sum(scores.values()) / len(scores):.2f}")
