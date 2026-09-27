import json
import os
import random

STATS_FILE = "data/rl_stats.json"

# Two different prompt "strategies" for extracting action items.
# The system will learn (via feedback) which one performs better over time.
STRATEGIES = {
    "strategy_a": """You are an assistant that extracts action items from meeting transcripts.
Read the transcript and extract all action items with task, assignee, and deadline.
Return ONLY a valid JSON array: [{{"task": "...", "assignee": "...", "deadline": "..."}}]

Transcript:
{transcript}
""",
    "strategy_b": """Carefully analyze this meeting transcript line by line.
Identify every commitment, task, or responsibility mentioned by any speaker.
For each one, extract: the task, who is responsible (assignee), and any deadline mentioned.
Return ONLY a valid JSON array: [{{"task": "...", "assignee": "...", "deadline": "..."}}]

Transcript:
{transcript}
""",
}

EPSILON = 0.2  # 20% chance to explore (try a random strategy), 80% to exploit (use the best one so far)


def load_stats():
    """Loads the performance stats of each strategy from file, or creates default stats."""
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "strategy_a": {"uses": 0, "positive_feedback": 0},
        "strategy_b": {"uses": 0, "positive_feedback": 0},
    }


def save_stats(stats):
    """Saves the updated stats back to file."""
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def choose_strategy():
    """
    Epsilon-greedy selection:
    - With probability EPSILON, pick a random strategy (exploration).
    - Otherwise, pick the strategy with the best success rate so far (exploitation).
    """
    stats = load_stats()

    if random.random() < EPSILON:
        chosen = random.choice(list(STRATEGIES.keys()))
    else:
        def success_rate(name):
            data = stats[name]
            if data["uses"] == 0:
                return 0.5  # neutral score if never used yet
            return data["positive_feedback"] / data["uses"]

        chosen = max(STRATEGIES.keys(), key=success_rate)

    # Record that this strategy was used
    stats[chosen]["uses"] += 1
    save_stats(stats)

    return chosen, STRATEGIES[chosen]


def record_feedback(strategy_name, was_helpful):
    """
    Updates the stats based on user feedback.
    was_helpful: True if the user marked the result as correct/good, False otherwise.
    """
    stats = load_stats()
    if was_helpful:
        stats[strategy_name]["positive_feedback"] += 1
    save_stats(stats)


def get_stats_summary():
    """Returns a readable summary of how each strategy is performing."""
    stats = load_stats()
    summary = {}
    for name, data in stats.items():
        rate = (data["positive_feedback"] / data["uses"] * 100) if data["uses"] > 0 else 0
        summary[name] = {
            "uses": data["uses"],
            "positive_feedback": data["positive_feedback"],
            "success_rate": f"{rate:.1f}%",
        }
    return summary