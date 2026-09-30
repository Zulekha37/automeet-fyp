from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def clean_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def validate_action_items(action_items):
    prompt = f"""You are a quality-control reviewer for a list of meeting action items.
Review the following JSON list and fix any issues you find:
- Merge or remove exact duplicate tasks.
- If an assignee is missing or unclear, set it to "Not specified" (never invent a name).
- If a deadline is not a real calendar date (e.g. September 31 does not exist), correct it to the nearest valid date or set it to "Not specified".
- Keep the priority field as given unless it clearly contradicts the deadline.

Action items:
{json.dumps(action_items, indent=2)}

Return ONLY the corrected JSON array, in the same format as the input, with no extra text.
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    try:
        cleaned = clean_json_response(response.choices[0].message.content)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return action_items


if __name__ == "__main__":
    from agents.extractor_agent import read_transcript, extract_action_items
    from agents.assignment_agent import assign_priority_and_dates

    transcript = read_transcript("data/sample_transcript.txt")
    raw_items = extract_action_items(transcript)
    assigned_items = assign_priority_and_dates(raw_items)

    print("Before validation:")
    print(json.dumps(assigned_items, indent=2))

    validated_items = validate_action_items(assigned_items)

    print("\nAfter validation:")
    print(json.dumps(validated_items, indent=2))