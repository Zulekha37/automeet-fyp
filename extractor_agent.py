from groq import Groq
from dotenv import load_dotenv
import os
import json

# Load API key from .env file
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def read_transcript(file_path):
    """Reads the transcript file"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_action_items(transcript_text):
    """Sends transcript to the LLM and extracts action items"""

    prompt = f"""
You are an assistant that extracts action items from meeting transcripts.

Read the transcript below and extract all action items. For each action item, identify:
1. The task description
2. The person responsible (assignee)
3. The deadline (if mentioned, otherwise say "Not specified")

Return ONLY a valid JSON array, with no extra text, in this exact format:
[
  {{"task": "task description", "assignee": "person name", "deadline": "deadline or Not specified"}}
]

Transcript:
{transcript_text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # Lower temperature for more consistent, accurate output
    )

    result_text = response.choices[0].message.content

    # Parse the JSON response
    try:
        action_items = json.loads(result_text)
        return action_items
    except json.JSONDecodeError:
        print("Warning: Response was not valid JSON. Raw response:")
        print(result_text)
        return []


if __name__ == "__main__":
    transcript = read_transcript("sample_transcript.txt")
    print("Transcript loaded. Extracting action items...\n")

    items = extract_action_items(transcript)

    print("=== Action Items ===")
    for i, item in enumerate(items, 1):
        print(f"\n{i}. Task: {item['task']}")
        print(f"   Assignee: {item['assignee']}")
        print(f"   Deadline: {item['deadline']}")