import streamlit as st
import pandas as pd
import json
from groq import Groq
from dotenv import load_dotenv
import os

from agents.assignment_agent import assign_priority_and_dates
from agents.rl_agent import choose_strategy, record_feedback, get_stats_summary

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="AutoMeet - Action Item Extractor", layout="wide")

st.title("AutoMeet: Meeting Action Item Extractor")
st.write("Paste your meeting transcript below and get organized action items automatically.")

# Text area for pasting transcript
transcript_input = st.text_area("Meeting Transcript", height=250, placeholder="Paste your meeting transcript here...")

if st.button("Extract Action Items"):
    if transcript_input.strip() == "":
        st.warning("Please paste a transcript first.")
    else:
        # The RL component picks which prompt strategy to use
        strategy_name, prompt_template = choose_strategy()
        st.session_state["strategy_used"] = strategy_name

        with st.spinner(f"Extracting action items (using {strategy_name})..."):
            prompt = prompt_template.format(transcript=transcript_input)
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            try:
                raw_items = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                raw_items = []

        if not raw_items:
            st.error("Could not extract action items. Please check the transcript format.")
        else:
            with st.spinner("Assigning priority and deadlines..."):
                final_items = assign_priority_and_dates(raw_items)

            st.session_state["final_items"] = final_items
            st.success(f"Found {len(final_items)} action items! (Strategy used: {strategy_name})")

# Show results if available
if "final_items" in st.session_state:
    df = pd.DataFrame(st.session_state["final_items"])
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", csv, "action_items.csv", "text/csv")

    st.write("### Was this result accurate?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Good Result"):
            record_feedback(st.session_state["strategy_used"], True)
            st.success("Thanks! The system will learn from this.")
    with col2:
        if st.button("Not Accurate"):
            record_feedback(st.session_state["strategy_used"], False)
            st.info("Thanks for the feedback! The system will learn from this.")

# Sidebar to show learning progress
with st.sidebar:
    st.write("### Learning Stats (RL Component)")
    stats = get_stats_summary()
    for strategy, data in stats.items():
        st.write(f"**{strategy}**")
        st.write(f"Used: {data['uses']} times")
        st.write(f"Success rate: {data['success_rate']}")
        st.write("---")