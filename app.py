import json
import os
import re
from typing import Dict, List, Union

import google.generativeai as palm
import numpy as np
import openai
import pandas as pd
import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from duckduckgo_search import DDGS
from PyPDF2 import PdfReader
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from streamlit_chat import message

from chatbot_models import *


_ = load_dotenv(find_dotenv())  # read local .env file


# Setting page title and header
st.set_page_config(page_title="WYN AI", page_icon=":robot_face:")
st.markdown(
    f"""
        <h1 style='text-align: center;'>Chatbot by Domain 🤖</h1>
    """,
    unsafe_allow_html=True,
)


# Initialise session state variables
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
# Check if "messages" is not stored in session state and set it to an empty list
if "messages" not in st.session_state:
    st.session_state.messages = []
# Iterate over each message in the session state messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if "domain_name" not in st.session_state:
    st.session_state["domain_name"] = []


# Sidebar - Instruction Manual
with st.sidebar:
    with st.expander("Instruction Manual 📖"):
        st.markdown(
            """
            ## WYN Chatbot

            Welcome to WYN Chatbot! 😄🤖

            ### Buttons

            1. **Similarity Algorithm** ✨

            Choose a similarity algorithm for text comparisons.

            - **Cosine Similarity** 🔄: Simple embedding based on word frequency count.
            - **OpenAI Embedding** 🌐: Utilizes OpenAI's embedding layer for semantic textual similarity.
            - **Palm Embedding** 🌴: Uses Palm's embedding layer for enhanced similarity calculations.
            - *And many more...*

            2. **Models** 🧠

            Select the chatbot model to power the conversation.

            - **ChatGPT 3** 🤖: Powerful language model for natural language understanding and generation.
            - **ChatGPT 4** 🤖🤖: Advanced version of ChatGPT 3 with improved capabilities.
            - **Palm** 🌴: Customized chatbot model using Palm's framework.
            - **Yin API** 🌐: Make API calls to Yiqiao Yin's custom "Yin" model via AWS API Gateway.
            - *And more model options available!*

            3. **Domains** 🌍

            Select a specific domain or topic for the chatbot to specialize in.

            - **Coder Domain** 💻: Request coding assistance or code generation.
            - **CBT Domain** 🧠💬: Experience Cognitive Behavioral Therapy with the chatbot acting as a therapist.
            - **PDF Scraping Domain** 📚🔍: Extract information from PDFs, such as financial reports or generative AI consulting reports.
            - *Various other domains available based on your preferences!*

            Please select your desired option by clicking on the corresponding button. Let's start chatting! 🎉
            """
        )


# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
similarity_indicator = st.sidebar.selectbox(
    "Choose a similarity algorithm:",
    ("Cosine", "Levenshtein", "STS", "STS-OpenAI", "STS-Palm", "Next..."),
)
model_name = st.sidebar.selectbox(
    "Choose a model:", ("GPT 3.5", "GPT 4", "Yin", "Palm", "Next...")
)
domain_name = st.sidebar.selectbox(
    "Choose a domain:",
    (
        "General",
        "CBT",
        "Coder",
        "Labcorp 2022 Annual Report",
        "Mckinsey Generative AI Report",
        "Adopting AI Responsibly",
        "Deep Learning Notes",
        "Upload Your Own",
    ),
)


# Load data
if domain_name == "Labcorp 2022 Annual Report":
    df = pd.read_csv("lh_ar_2022.csv")
elif domain_name == "Mckinsey Generative AI Report":
    df = pd.read_csv("mckinsey_gen_ai.csv")
elif domain_name == "Adopting AI Responsibly":
    df = pd.read_csv("adopt_ai_responsibly.csv")
else:
    df = pd.DataFrame()
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Next item ... ")
clear_button = st.sidebar.button("Clear Conversation", key="clear")
st.sidebar.markdown(
    "@ [Yiqiao Yin](https://www.y-yin.io/) | [LinkedIn](https://www.linkedin.com/in/yiqiaoyin/) | [YouTube](https://youtube.com/YiqiaoYin/)"
)


# Reset everything
if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.session_state["number_tokens"] = []
    st.session_state["domain_name"] = []
    counter_placeholder.write(f"Next item ...")


# openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_key = st.secrets["OPENAI_API_KEY"]


def call_yin_test1(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        # Handle any potential errors
        return None


# palm_api_key = os.environ['PALM_API_KEY']
palm_api_key = st.secrets["PALM_API_KEY"]
palm.configure(api_key=palm_api_key)


if domain_name == "Upload Your Own":
    st.write("### Upload or select your PDF file")
    uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")

    pdf_content = None
    if uploaded_file is not None:
        pdf_content = extract_data(uploaded_file)
        st.success("Page extraction completed.")


# container for chat history
response_container = st.container()
# container for text box
container = st.container()


# Get user input from chat_input and store it in the prompt variable using the walrus operator ":="
if prompt := st.chat_input("What is up?"):
    # Add user message to session state messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if domain_name == "General":
        processed_user_question = f"""
            You are an AI assistant for the user.
            Answer the following question from the user: {user_input}
        """
        if model_name == "GPT 3.5":
            output = call_chatgpt(processed_user_question)
        elif model_name == "Palm":
            output = call_palm(processed_user_question)
        elif model_name == "Yin":
            query = processed_user_question
            api_url = f"https://y3q3szoxua.execute-api.us-east-1.amazonaws.com/dev/my-openai-api-test1?query={query}&key={user_key}"
            output = call_yin_test1(api_url)["answer"]
        else:
            output = call_chatgpt(processed_user_question)
    elif domain_name == "Coder":
        processed_user_question = f"""
            You are an AI assistant for the user.
            Answer the following question from the user: {user_input}
        """
        if model_name == "GPT 3.5":
            output = call_chatgpt(processed_user_question)
        elif model_name == "Palm":
            output = call_palm(processed_user_question)
        elif model_name == "Yin":
            query = processed_user_question
            api_url = f"https://y3q3szoxua.execute-api.us-east-1.amazonaws.com/dev/my-openai-api-test1?query={query}&key={user_key}"
            output = call_yin_test1(api_url)["answer"]
        else:
            output = call_chatgpt(processed_user_question)
    elif domain_name in [
        "Labcorp 2022 Annual Report",
        "Mckinsey Generative AI Report",
        "Adopting AI Responsibly",
    ]:
        df_screened_by_dist_score = add_dist_score_column(
            df, user_input, similarity_indicator.lower().replace("-", "")
        )
        qa_pairs = convert_to_list_of_dict(df_screened_by_dist_score)
        qa_pairs_single = convert_to_list_of_dict_single_pair(
            df_screened_by_dist_score
        )

        processed_user_question = f"""
            Learn from the context: {qa_pairs}
            Answer the following question as if you are the AI assistant: {user_input}
            Produce a text answer that are complete sentences.
        """
        if model_name == "GPT 3.5":
            output = call_chatgpt(processed_user_question)
        elif model_name == "GPT 4":
            output = call_chatcompletion(messages=qa_pairs)
        elif model_name == "Palm":
            output = call_palm(processed_user_question)
        else:
            output = call_chatgpt(processed_user_question)
    elif domain_name == "CBT":
        processed_user_question = f"""
            You are therapist for the user. You specialize in Cognitive Behavioral Therapy.
            Answer the following question from the user: {user_input}
            Make sure to be aware of suicidal symptoms, depression, anxiety disorders.
            Be patient with the user and try to comfort them.
        """
        if model_name == "GPT 3.5":
            output = call_chatgpt(processed_user_question)
        elif model_name == "Palm":
            output = call_palm(processed_user_question)
        else:
            output = call_chatgpt(processed_user_question)
    elif domain_name == "Upload Your Own":
        processed_user_question = f"""
            Learn from the context: {pdf_content}
            Answer the following question as if you are the AI assistant: {user_input}
            Produce a text answer that are complete sentences.
        """
        if model_name == "GPT 3.5":
            output = call_chatgpt(processed_user_question)
        elif model_name == "Palm":
            output = call_palm(processed_user_question)
        else:
            output = call_chatgpt(processed_user_question)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(output)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": output})
