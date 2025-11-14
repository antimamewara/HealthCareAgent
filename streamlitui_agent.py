# app.py
import streamlit as st
import requests , os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not set in environment")


client = OpenAI(api_key=OPENAI_API_KEY)


# streamlit_app.py
# import streamlit as st
# import requests, os

# st.title("Clinical Summary Agent")
# patient_id = st.text_input("Patient ID")
# question = st.text_area("Ask about patient history / diagnosis")
# user_id = st.text_input("Your doctor ID")

# if st.button("Get Summary"):
#     if not patient_id or not question or not user_id:
#         st.warning("Provide patient_id, question and your user id")
#     else:
#         with st.spinner("Fetching summary..."):
#             resp = requests.post(
#                 os.getenv("API_URL") + "/ask",
#                 json={"patient_id":patient_id, "question":question, "user_id":user_id},
#                 timeout=60
#             )
#             if resp.status_code==200:
#                 data = resp.json()["result"]
#                 st.json(data)
#             else:
#                 st.error(f"Error: {resp.text}")


st.set_page_config(page_title="AI Agent", page_icon="🤖")
st.title("🤖 Multi-Agent LLM App")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Streaming response from OpenAI (can replace with your agent)
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
