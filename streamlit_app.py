import streamlit as st
import pandas as pd
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face API
HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Load dataset
df = pd.read_csv("biscayneBay_waterquality.csv")

# Streamlit UI
st.title("💧 Water Quality AI Assistant")
st.write("Ask questions about the water quality dataset!")

# Allow user to enter API key
hf_key_input = st.text_input("Enter Hugging Face API Key", type="password")

if hf_key_input:
    headers = {"Authorization": f"Bearer {hf_key_input}"}

# Preview dataset
st.subheader("Dataset Preview")
st.dataframe(df)

st.write("Dataset Columns:", df.columns)
tab1 = st.tabs(
    ["Descriptive Statistics"]
)

# ------------------- TAB 1 -------------------
with tab1:
    st.info("Dataset Overview")
    st.dataframe(df)
    st.caption("Raw Data")

    st.divider()

    st.dataframe(df.describe())
    st.caption("Descriptive Statistics")


# Retrieve relevant data
def retrieve_relevant_data(question, dataframe):

    question = question.lower()

    if "ph" in question and "pH" in dataframe.columns:
        return dataframe[["pH"]]

    elif "turbidity" in question and "turbidity" in dataframe.columns:
        return dataframe[["turbidity"]]

    elif "oxygen" in question and "dissolved_oxygen" in dataframe.columns:
        return dataframe[["dissolved_oxygen"]]

    else:
        return dataframe.head(10)


# Hugging Face query
def query_huggingface(prompt):

    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)

    return response.json()


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
if prompt := st.chat_input("Ask a question about the water data"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve dataset rows
    relevant_data = retrieve_relevant_data(prompt, df)

    context = relevant_data.to_string(index=False)

    ai_prompt = f"""
You are a water quality expert.

Use ONLY the dataset below to answer the question.

Dataset:
{context}

Question:
{prompt}

If the answer is not in the dataset, say the dataset does not contain the answer.
"""

    result = query_huggingface(ai_prompt)

    try:
        reply = result[0]["generated_text"]
    except:
        reply = "Model could not generate a response."

    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
