from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from groq import Groq
from dotenv import load_dotenv
import os
import json
from datetime import date

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class MeetingState(TypedDict):
    transcript: str
    raw_items: List[Dict]
    final_items: List[Dict]


def clean_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def extractor_node(state: MeetingState) -> MeetingState:
    prompt = f"""You are an assistant that extracts action items from meeting transcripts.
Read the transcript and extract all action items with task, assignee, and deadline.
Return ONLY a valid JSON array: [{{"task": "...", "assignee": "...", "deadline": "..."}}]

Transcript:
{state['transcript']}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    try:
        cleaned = clean_json_response(response.choices[0].message.content)
        items = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Extractor JSON error: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        items = []

    return {"raw_items": items}


def assignment_node(state: MeetingState) -> MeetingState:
    today = date.today().strftime("%Y-%m-%d")

    prompt = f"""Today's date is {today}.
Given these action items, convert each deadline into an actual date (YYYY-MM-DD)
and assign a priority (High/Medium/Low) based on urgency.

Action items:
{json.dumps(state['raw_items'], indent=2)}

Return ONLY a valid JSON array:
[{{"task": "...", "assignee": "...", "deadline": "YYYY-MM-DD", "priority": "High/Medium/Low"}}]
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    try:
        cleaned = clean_json_response(response.choices[0].message.content)
        final_items = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Assignment JSON error: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        final_items = []

    return {"final_items": final_items}


def output_node(state: MeetingState) -> MeetingState:
    with open("data/action_items_output.json", "w", encoding="utf-8") as f:
        json.dump(state["final_items"], f, indent=2)
    return {}


builder = StateGraph(MeetingState)
builder.add_node("extractor", extractor_node)
builder.add_node("assignment", assignment_node)
builder.add_node("output", output_node)

builder.add_edge(START, "extractor")
builder.add_edge("extractor", "assignment")
builder.add_edge("assignment", "output")
builder.add_edge("output", END)

graph = builder.compile()


if __name__ == "__main__":
    with open("data/sample_transcript.txt", "r", encoding="utf-8") as f:
        transcript = f.read()

    print("Running the LangGraph pipeline...\n")
    result = graph.invoke({"transcript": transcript, "raw_items": [], "final_items": []})

    print("=== Final Result ===")
    print(json.dumps(result["final_items"], indent=2))