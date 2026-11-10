
"""Streamlit interface for the Enterprise Knowledge Platform."""

import streamlit as st

from src.agent.agent import ask_agent


st.set_page_config(
    page_title="Enterprise Knowledge Platform",
    page_icon="📚",
    layout="wide",
)

st.title("Enterprise Knowledge & Workflow Intelligence")
st.caption(
    "Ask questions about Azure documentation, "
    "analyze sales data, or perform numerical calculations."
)

# Keep the conversation in session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept a new question.
if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Working on your question..."):
            try:
                answer = ask_agent(prompt)
            except Exception:
                st.error(
                    "The request failed. Check the application "
                    "logs for details."
                )
                answer = None

        if answer is not None:
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
