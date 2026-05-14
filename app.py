"""
Streamlit UI for the Target Prioritisation Agent.

Run with:  streamlit run app.py

This is intentionally simple — the interesting work happens in the agent.
The UI just provides a text input, a run button, and displays the result.
For a portfolio project, a clean UI that gets out of the way is better
than a flashy one that distracts from the science.
"""

import streamlit as st
from src.agent import run_agent
from src.config import GOOGLE_API_KEY


def main():
    st.set_page_config(
        page_title="Target Prioritisation Agent",
        page_icon="🧬",
        layout="wide",
    )

    st.title("🧬 Target Prioritisation Agent")
    st.markdown(
        "Enter a disease indication to identify and rank potential drug targets. "
        "The agent queries **OpenTargets** for genetic associations and **GTEx** "
        "for tissue expression, then synthesises a ranked assessment using an LLM."
    )

    # Warn if API key is missing — common setup issue
    if not GOOGLE_API_KEY:
        st.error(
            "⚠️ GOOGLE_API_KEY not found.  "
            "Copy `.env.example` to `.env` and add your key from https://aistudio.google.com/apikey"
        )
        st.stop()

    # Main input
    disease_name = st.text_input(
        "Disease indication",
        placeholder="e.g., ulcerative colitis, acute myeloid leukemia, Crohn's disease",
    )

    # Example buttons to make the demo faster during an interview
    st.markdown("**Quick examples:**")
    cols = st.columns(4)
    examples = [
        "acute myeloid leukemia",
        "ulcerative colitis",
        "non-small cell lung cancer",
        "rheumatoid arthritis",
    ]
    for col, example in zip(cols, examples):
        if col.button(example, use_container_width=True):
            disease_name = example

    if disease_name:
        with st.spinner(
            f"Agent is analysing targets for **{disease_name}**... "
            f"This takes 30-60 seconds as the agent makes multiple API calls."
        ):
            try:
                result = run_agent(disease_name)
                st.markdown("---")
                st.markdown("### Agent Analysis")
                st.markdown(result)
            except Exception as e:
                st.error(f"Agent failed: {e}")
                st.info(
                    "This usually means an API is temporarily down. "
                    "Try again in a minute, or check the console for details."
                )


if __name__ == "__main__":
    main()
