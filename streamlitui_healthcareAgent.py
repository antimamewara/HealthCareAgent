# app.py
import streamlit as st
import requests , os
from dotenv import load_dotenv

load_dotenv()

# streamlit_app.py
import streamlit as st
import requests, os

st.title("Clinical Summary Agent")
patient_id = st.text_input("Patient ID")
question = st.text_area("Ask about patient history / diagnosis")
user_id = st.text_input("Your doctor ID")

if st.button("Get Summary"):
    if not patient_id or not question or not user_id:
        st.warning("Provide patient_id, question and your user id")
    else:
        with st.spinner("Fetching summary..."):
            resp = requests.post(
                os.getenv("API_URL") + "/ask",
                json={"patient_id":patient_id, "question":question, "user_id":user_id},
                timeout=60
            )
            if resp.status_code==200:
                data = resp.json()["result"][1]
                st.json(data)
            else:
                st.error(f"Error: {resp.text}")

