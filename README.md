# AutoMeet

**AutoMeet** is an AI-powered multi-agent system that automatically extracts action items from meeting transcripts. It identifies tasks, assigns owners and deadlines, prioritizes them, and presents the results through a simple web dashboard — removing the need for manual note-taking and follow-up after meetings.

This project was built as a Final Year Project under the domain **Agentic AI and Intelligent Systems**.

## Features

- Extracts action items (task, assignee, deadline) from raw meeting transcripts
- Converts relative deadlines (e.g. "by Friday") into actual calendar dates
- Assigns a priority level (High / Medium / Low) to each task
- Multi-agent pipeline built with **CrewAI**
- Reinforcement-learning component (epsilon-greedy) that improves extraction accuracy from user feedback
- Simple web dashboard built with **Streamlit**
- Alternative pipeline implementation using **LangGraph**

## Project Structure

```
automeet-fyp/
├── agents/
│   ├── __init__.py
│   ├── extractor_agent.py     # Extracts tasks, assignees, deadlines
│   ├── assignment_agent.py    # Assigns dates and priority
│   ├── output_agent.py        # Formats and saves final output
│   ├── crew_agents.py         # CrewAI multi-agent pipeline
│   ├── langgraph_agents.py    # LangGraph pipeline (alternative implementation)
│   └── rl_agent.py            # Epsilon-greedy learning component
├── data/
│   ├── sample_transcript.txt
│   ├── rl_stats.json
│   └── action_items_output.json
├── dashboard.py                # Streamlit web interface
├── test.py                     # API connectivity check
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```

4. Run the dashboard:
   ```
   python -m streamlit run dashboard.py
   ```

## Running the Agent Pipelines Directly

CrewAI version:
```
python agents/crew_agents.py
```

LangGraph version:
```
python agents/langgraph_agents.py
```

## Team

- Muneeb
- Zulekha
- Ayesha

## Documentation

See the project proposal, PRD, and TRD documents for full details on objectives, requirements, and technical design.