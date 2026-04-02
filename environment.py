from pydantic import BaseModel
from typing import Optional
import random

# ─── Data Models ───────────────────────────────────────────

class Observation(BaseModel):
    document: str
    document_type: str        # "job_post" or "news_article"
    task_id: str
    instructions: str

class Action(BaseModel):
    bias_phrases: list[str]           # list of biased phrases found
    ai_sentences: list[str]           # list of AI-generated sentences found
    severity: str                     # "low", "medium", "high"
    explanation: str                  # why you flagged these

class Reward(BaseModel):
    score: float
    feedback: str

# ─── Dataset ───────────────────────────────────────────────

DOCUMENTS = {
    "easy": {
        "id": "easy_001",
        "type": "job_post",
        "text": """
            We are looking for a young, energetic man to join our 
            dynamic startup team. The ideal candidate is a recent 
            graduate who is hungry for success. 
            Native English speakers only. Must be physically fit.
        """,
        "bias_markers": ["young", "man", "native english speakers only", "physically fit"],
        "ai_sentences": [],
        "severity": "high"
    },
    "medium": {
        "id": "medium_001",
        "type": "news_article",
        "text": """
            The company announced record profits today. 
            In conclusion, it is important to note that 
            the implications of this development are multifaceted 
            and far-reaching. Furthermore, stakeholders should 
            consider the broader context of these findings.
            The CEO was pleased with the results.
        """,
        "bias_markers": [],
        "ai_sentences": [
            "In conclusion, it is important to note that the implications of this development are multifaceted and far-reaching.",
            "Furthermore, stakeholders should consider the broader context of these findings."
        ],
        "severity": "medium"
    },
    "hard": {
        "id": "hard_001",
        "type": "job_post",
        "text": """
            We are seeking a young and dynamic female receptionist 
            for our front office. It is worth noting that this role 
            requires someone who is presentable and well-spoken. 
            Furthermore, the ideal candidate should be able to 
            multitask effectively in a fast-paced environment.
            No hijab or religious accessories allowed.
        """,
        "bias_markers": ["young", "female", "presentable", "no hijab or religious accessories allowed"],
        "ai_sentences": [
            "It is worth noting that this role requires someone who is presentable and well-spoken.",
            "Furthermore, the ideal candidate should be able to multitask effectively in a fast-paced environment."
        ],
        "severity": "high"
    }
}

# ─── Graders ───────────────────────────────────────────────

def grade_easy(action: Action, doc: dict) -> tuple[float, str]:
    found = [p.lower() for p in action.bias_phrases]
    markers = doc["bias_markers"]
    hits = sum(1 for m in markers if any(m in f for f in found))
    score = round(hits / len(markers), 2) if markers else 0.0
    return score, f"Found {hits}/{len(markers)} bias markers"

def grade_medium(action: Action, doc: dict) -> tuple[float, str]:
    found = [s.lower() for s in action.ai_sentences]
    targets = [s.lower() for s in doc["ai_sentences"]]
    hits = sum(1 for t in targets if any(t[:20] in f for f in found))
    score = round(hits / len(targets), 2) if targets else 0.0
    return score, f"Found {hits}/{len(targets)} AI-generated sentences"

def grade_hard(action: Action, doc: dict) -> tuple[float, str]:
    bias_score, _ = grade_easy(action, doc)
    ai_score, _ = grade_medium(action, doc)
    severity_score = 0.2 if action.severity == doc["severity"] else 0.0
    total = round((bias_score * 0.4) + (ai_score * 0.4) + severity_score, 2)
    return total, f"Bias: {bias_score}, AI: {ai_score}, Severity: {severity_score}"

GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard
}

# ─── Environment ───────────────────────────────────────────

class TruthHireEnv:

    def __init__(self):
        self.current_task = None
        self.step_count = 0
        self.max_steps = 5
        self.last_reward = None

    def reset(self, task_id: str = "easy") -> Observation:
        self.current_task = DOCUMENTS[task_id]
        self.step_count = 0
        self.last_reward = None
        return Observation(
            document=self.current_task["text"],
            document_type=self.current_task["type"],
            task_id=task_id,
            instructions=(
                "Analyze the document. Find any biased phrases, "
                "AI-generated sentences, rate severity (low/medium/high), "
                "and explain your findings."
            )
        )

    def step(self, action: Action):
        self.step_count += 1
        grader = GRADERS[self.current_task["id"].split("_")[0]]
        score, feedback = grader(action, self.current_task)
        self.last_reward = Reward(score=score, feedback=feedback)
        done = self.step_count >= self.max_steps or score >= 0.8
        return (
            Observation(
                document=self.current_task["text"],
                document_type=self.current_task["type"],
                task_id=self.current_task["id"],
                instructions="Refine your analysis if needed."
            ),
            self.last_reward,
            done,
            {"step": self.step_count}
        )

    def state(self) -> dict:
        return {
            "current_task": self.current_task["id"] if self.current_task else None,
            "step_count": self.step_count,
            "last_score": self.last_reward.score if self.last_reward else None
        }