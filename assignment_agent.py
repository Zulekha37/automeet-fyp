from groq import Groq
from dotenv import load_dotenv
import os
import json
from datetime import date

# Import function from extractor agent
from extractor_agent import read_transcript, extract_action_items

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def assign_priority_and_dates(action_items):
    """Assigns priority to each action item and converts deadline into an actual date"""

    today = date.today().strftime("%Y-%m-%d")

    prompt = f"""
Today's date is {today}.

You are given a list of action items from a meeting. For each item:
1. Convert the deadline into an actual date in YYYY-MM-DD format (calculate based on today's date). If no deadline was given, use "Not specified".
2. Assign a priority: "High" if the deadline is within 3 days, "Medium" if within a week, "Low" if more than a week or not specified.

Action items:
{json.dumps(action_items, indent=2)}

Return ONLY a valid JSON array in this exact format, with no extra text:
[
  {{"task": "task description", "assignee": "person name", "deadline": "YYYY-MM-DD or Not specified", "priority": "High/Medium/Low"}}
]
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    result_text = response.choices[0].message.content

    try:
        final_items = json.loads(result_text)
        return final_items
    except json.JSONDecodeError:
        print("Warning: Response was not valid JSON. Raw response:")
        print(result_text)
        return []


if __name__ == "__main__":
    transcript = read_transcript("sample_transcript.txt")
    print("Transcript loaded. Extracting action items...\n")

    raw_items = extract_action_items(transcript)
    print(f"{len(raw_items)} action items found. Assigning priority and dates...\n")

    final_items = assign_priority_and_dates(raw_items)

    print("=== Final Action Items (with Priority & Dates) ===")
    for i, item in enumerate(final_items, 1):
        print(f"\n{i}. Task: {item['task']}")
        print(f"   Assignee: {item['assignee']}")
        print(f"   Deadline: {item['deadline']}")
        print(f"   Priority: {item['priority']}")