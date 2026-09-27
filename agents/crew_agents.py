from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
import os
import litellm

# Fix for a known compatibility issue between CrewAI and Groq
_original_completion = litellm.completion

def _patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
    return _original_completion(*args, **kwargs)

litellm.completion = _patched_completion

load_dotenv()

groq_llm = LLM(
    model="groq/openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

extractor_agent = Agent(
    role="Meeting Transcript Analyst",
    goal="Extract all action items, assignees, and deadlines from a meeting transcript",
    backstory="You are an expert at carefully reading meeting transcripts and identifying every task that was assigned, who it was assigned to, and any deadline mentioned.",
    llm=groq_llm,
    verbose=True,
)

assignment_agent = Agent(
    role="Task Prioritization Specialist",
    goal="Convert deadlines into actual dates and assign a priority level to each task",
    backstory="You are skilled at time management and understand how to judge urgency based on deadlines.",
    llm=groq_llm,
    verbose=True,
)

output_agent = Agent(
    role="Report Formatter",
    goal="Organize the final action items into a clean, structured format",
    backstory="You specialize in presenting information clearly and concisely for business reports.",
    llm=groq_llm,
    verbose=True,
)


def run_crew(transcript_text):
    extract_task = Task(
        description=f"""Read the following meeting transcript and extract all action items.
For each one, identify the task, the assignee, and the deadline (if mentioned).

Transcript:
{transcript_text}
""",
        expected_output="A list of action items, each with a task description, assignee, and deadline.",
        agent=extractor_agent,
    )

    assign_task = Task(
        description="""Take the extracted action items and:
1. Convert each deadline into an actual date (assume today's date if needed).
2. Assign a priority: High, Medium, or Low, based on urgency.
""",
        expected_output="A list of action items with task, assignee, deadline (as a date), and priority.",
        agent=assignment_agent,
        context=[extract_task],
    )

    format_task = Task(
        description="""Take the final action items and format them into a clean JSON array with the fields:
task, assignee, deadline, priority. Return ONLY the JSON array, nothing else.
""",
        expected_output="A valid JSON array of action items.",
        agent=output_agent,
        context=[assign_task],
    )

    crew = Crew(
        agents=[extractor_agent, assignment_agent, output_agent],
        tasks=[extract_task, assign_task, format_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return result


if __name__ == "__main__":
     with open("data/sample_transcript.txt", "r", encoding="utf-8") as file:
        transcript = file.read()

    print("Running the crew...\n")
    result = run_crew(transcript)

    print("\n=== Final Result ===")
    print(result)