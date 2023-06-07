import json
from typing import Dict, List, Union

import numpy as np
import openai
import os
import pandas as pd
import streamlit as st
import google.generativeai as palm
from scipy.spatial.distance import cosine

from sentence_transformers import SentenceTransformer
from streamlit_chat import message

# Setting page title and header
st.set_page_config(page_title="WYN AI", page_icon=":robot_face:")
st.markdown(
    f"""
        <h1 style='text-align: center;'>W.Y.N. Artificial Intelligence😬</h1>
    """,
    unsafe_allow_html=True,
)

# Load data
df = pd.read_csv("lh_ar_2022.csv")

# Initialise session state variables
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if "domain_name" not in st.session_state:
    st.session_state["domain_name"] = []


# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio(
    "Choose a model:", ("ChatGPT", "Palm", "Next...")
)
domain_name = st.sidebar.radio(
    "Choose a domain:", ("General", "Labcorp 2022 Annual Report", "Upload Your Own")
)
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Next item ... ")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# reset everything
if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state["number_tokens"] = []
    st.session_state["domain_name"] = []
    counter_placeholder.write(f"Next item ...")

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
def call_chatgpt(prompt: str) -> str:
    """
    Uses the OpenAI API to generate an AI response to a prompt.

    Args:
        prompt: A string representing the prompt to send to the OpenAI API.

    Returns:
        A string representing the AI's generated response.

    """

    # Use the OpenAI API to generate a response based on the input prompt.
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Extract the text from the first (and only) choice in the response output.
    ans = response.choices[0]["text"]

    # Return the generated AI response.
    return ans


palm_api_key  = os.environ['PALM_API_KEY']
palm.configure(api_key=palm_api_key)

def call_palm(prompt: str) -> str:
    completion = palm.generate_text(
        model="models/text-bison-001",
        prompt=prompt,
        temperature=0,
        max_output_tokens=800,
    )

    return completion.result

def calculate_cosine_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the cosine similarity between two sentences.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The cosine similarity between the two sentences, represented as a float value between 0 and 1.
    """
    # Tokenize the sentences into words
    words1 = sentence1.lower().split()
    words2 = sentence2.lower().split()

    # Create a set of unique words from both sentences
    unique_words = set(words1 + words2)

    # Create a frequency vector for each sentence
    freq_vector1 = np.array([words1.count(word) for word in unique_words])
    freq_vector2 = np.array([words2.count(word) for word in unique_words])

    # Calculate the cosine similarity between the frequency vectors
    similarity = 1 - cosine(freq_vector1, freq_vector2)

    return similarity


def levenshtein_distance(s1: str, s2: str) -> float:
    """
    Compute the Levenshtein distance between two strings.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        float: The Levenshtein distance between the two strings.
    """
    m = len(s1)
    n = len(s2)

    # Create a matrix to store the distances between substrings of s1 and s2
    d = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column of the matrix
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    # Compute the distances between all substrings of s1 and s2
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1

    # Return the Levenshtein distance between the two strings
    return d[m][n] * (-1)


def calculate_sts_score(sentence1: str, sentence2: str) -> float:
    model = SentenceTransformer(
        "paraphrase-MiniLM-L6-v2"
    )  # Load a pre-trained STS model

    # Compute sentence embeddings
    embedding1 = model.encode([sentence1])[0]  # Flatten the embedding array
    embedding2 = model.encode([sentence2])[0]  # Flatten the embedding array

    # Calculate cosine similarity between the embeddings
    similarity_score = 1 - cosine(embedding1, embedding2)

    return similarity_score


def add_dist_score_column(
    dataframe: pd.DataFrame, sentence: str, similarity_indicator: str = "cosine"
) -> pd.DataFrame:
    if similarity_indicator == "cosine":
        dataframe["cosine"] = dataframe["questions"].apply(
            lambda x: calculate_cosine_similarity(x, sentence)
        )
    elif similarity_indicator == "levenshtein":
        dataframe["levenshtein"] = dataframe["questions"].apply(
            lambda x: calculate_cosine_similarity(x, sentence)
        )
    elif similarity_indicator == "sts":
        dataframe["sts"] = dataframe["questions"].apply(
            lambda x: calculate_sts_score(x, sentence)
        )
    else:
        dataframe["cosine"] = dataframe["questions"].apply(
            lambda x: calculate_cosine_similarity(x, sentence)
        )

    sorted_dataframe = dataframe.sort_values(by=similarity_indicator, ascending=False)

    return sorted_dataframe.iloc[:5, :]


def add_dist_score_column(
    dataframe: pd.DataFrame, sentence: str, similarity_indicator: str = "cosine"
) -> pd.DataFrame:
    if similarity_indicator == "cosine":
        dataframe["cosine"] = dataframe["questions"].apply(
            lambda x: calculate_cosine_similarity(x, sentence)
        )
    elif similarity_indicator == "levenshtein":
        dataframe["levenshtein"] = dataframe["questions"].apply(
            lambda x: calculate_cosine_similarity(x, sentence)
        )
    elif similarity_indicator == "sts":
        dataframe["sts"] = dataframe["questions"].apply(
            lambda x: calculate_sts_score(x, sentence)
        )
    else:
        dataframe["cosine"] = dataframe["questions"].apply(
            lambda x: calculate_cosine_similarity(x, sentence)
        )

    sorted_dataframe = dataframe.sort_values(by=similarity_indicator, ascending=False)

    return sorted_dataframe.iloc[:5, :]


def convert_to_list_of_dict(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Reads in a pandas DataFrame and produces a list of dictionaries with two keys each, 'question' and 'answer.'

    Args:
        df: A pandas DataFrame with columns named 'questions' and 'answers'.

    Returns:
        A list of dictionaries, with each dictionary containing a 'question' and 'answer' key-value pair.
    """

    # Initialize an empty list to store the dictionaries
    result = []

    # Loop through each row of the DataFrame
    for index, row in df.iterrows():
        # Create a dictionary with the current question and answer
        qa_dict_quest = {"role": "user", "content": row["questions"]}
        qa_dict_ans = {"role": "assistant", "content": row["answers"]}

        # Add the dictionary to the result list
        result.append(qa_dict_quest)
        result.append(qa_dict_ans)

    # Return the list of dictionaries
    return result


from PyPDF2 import PdfReader


def extract_data(feed):
    pdf_reader = PdfReader(feed)
    pages = len(pdf_reader.pages)

    text_list = []
    for page in range(pages):
        pdf_page = pdf_reader.pages[page]
        text = pdf_page.extract_text()
        text_list.append(text)

    return text


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

with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        if domain_name == "General":
            processed_user_question = f"""
                You are an AI assistant for the user.
                Answer the following question from the user: {user_input}
            """
            if model_name == "ChatGPT":
                output = call_chatgpt(processed_user_question)
            elif model_name == "Palm":
                output = call_palm(processed_user_question)
            else:
                output = call_chatgpt(processed_user_question)

        elif domain_name == "Labcorp 2022 Annual Report":
            df_screened_by_dist_score = add_dist_score_column(df, user_input, "cosine")
            qa_pairs = convert_to_list_of_dict(df_screened_by_dist_score)

            processed_user_question = f"""
                Learn from the context: {qa_pairs}
                Answer the following question as if you are the AI assistant: {user_input}
                Produce a text answer that are complete sentences.
            """
            if model_name == "ChatGPT":
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
            if model_name == "ChatGPT":
                output = call_chatgpt(processed_user_question)
            elif model_name == "Palm":
                output = call_palm(processed_user_question)
            else:
                output = call_chatgpt(processed_user_question)

        # update session
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))
            counter_placeholder.write(f"All rights reserved @ Yiqiao Yin")