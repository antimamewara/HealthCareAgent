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
        url = os.getenv("API_URL") + "/ask"
        headers = {"Content-Type": "application/json"}
        payload = {"patient_id":patient_id, "question":question, "user_id":user_id}
        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=60
        )
        if resp.status_code == 200:
            try:
                j = resp.json()
            except ValueError:
                st.error("Response is not valid JSON")
            else:
                result = j.get("result")
                # handle common shapes: list with items, dict, or plain string
                if isinstance(result, (list, tuple)):
                    if len(result) > 1 and result[1] is not None:
                        st.json(result[1])
                    elif len(result) > 0:
                        st.json(result[0])
                    else:
                        st.info("Response 'result' is an empty list")
                elif isinstance(result, dict):
                    st.json(result)
                elif isinstance(result, str):
                    st.text(result)
                else:
                    # fallback: show whole JSON so you can inspect it
                    st.write("Unexpected response shape; full JSON:")
                    st.json(j)
        else:
            st.error(f"HTTP {resp.status_code}: {resp.text}")

