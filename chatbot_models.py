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
import openai
from openai import OpenAI
from typing import List, Dict, Any


# palm_api_key = os.environ['PALM_API_KEY']
palm_api_key = st.secrets["PALM_API_KEY"]
palm.configure(api_key=palm_api_key)


openai.api_key = st.secrets["OPENAI_API_KEY"]
openai_client = OpenAI()


def call_chatgpt(query: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Generates a response to a query using the specified language model.

    Args:
        query (str): The user's query that needs to be processed.
        model (str, optional): The language model to be used. Defaults to "gpt-3.5-turbo".

    Returns:
        str: The generated response to the query.
    """

    # Prepare the conversation context with system and user messages.
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question: {query}."},
    ]

    # Use the OpenAI client to generate a response based on the model and the conversation context.
    response: Any = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # Extract the content of the response from the first choice.
    content: str = response.choices[0].message.content

    # Return the generated content.
    return content


def call_chatcompletion(
    messages: list, model: str = "gpt-4", temperature: int = 0
) -> str:
    """
    Get the completion response from a list of messages using OpenAI's ChatCompletion API.

    Parameters:
    - messages (list): A list of messages which includes role ("user" or "assistant") and content.
    - model (str): The name of the OpenAI model to use. Default is "gpt-4".
    - temperature (int): The temperature parameter for generating more random or deterministic responses. Default is 0.

    Returns:
    - str: The content of the first response choice in the completed message.

    """
    # Call OpenAI's ChatCompletion API with the specified parameters
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=4000,
    )

    # Get the content of the first response choice in the completed message
    return response.choices[0].message["content"]


def call_yin_test1(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        # Handle any potential errors
        return None


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


def openai_text_embedding(prompt: str) -> str:
    return openai.Embedding.create(input=prompt, model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]


def calculate_sts_openai_score(sentence1: str, sentence2: str) -> float:
    # Compute sentence embeddings
    embedding1 = openai_text_embedding(sentence1)  # Flatten the embedding array
    embedding2 = openai_text_embedding(sentence2)  # Flatten the embedding array

    # Convert to array
    embedding1 = np.asarray(embedding1)
    embedding2 = np.asarray(embedding2)

    # Calculate cosine similarity between the embeddings
    similarity_score = 1 - cosine(embedding1, embedding2)

    return similarity_score


def palm_text_embedding(prompt: str) -> str:
    model = "models/embedding-gecko-001"
    return palm.generate_embeddings(model=model, text=prompt)["embedding"]


def calculate_sts_palm_score(sentence1: str, sentence2: str) -> float:
    # Compute sentence embeddings
    embedding1 = palm_text_embedding(sentence1)  # Flatten the embedding array
    embedding2 = palm_text_embedding(sentence2)  # Flatten the embedding array

    # Convert to array
    embedding1 = np.asarray(embedding1)
    embedding2 = np.asarray(embedding2)

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
    elif similarity_indicator == "stsopenai":
        dataframe["stsopenai"] = dataframe["questions"].apply(
            lambda x: calculate_sts_openai_score(str(x), sentence)
        )
    elif similarity_indicator == "stspalm":
        dataframe["stspalm"] = dataframe["questions"].apply(
            lambda x: calculate_sts_palm_score(str(x), sentence)
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


def convert_to_list_of_dict_single_pair(df: pd.DataFrame) -> List[Dict[str, str]]:
    questions = pd.DataFrame()
    answers = pd.DataFrame()

    for i in range(df.shape[0]):
        processed_df_quest = pd.DataFrame()
        processed_df_ans = pd.DataFrame()

        s = df.iloc[i, 2]
        split_s = re.split(r"\d+\.", s)
        split_s = [i.strip() for i in split_s if i]
        processed_df_quest["questions"] = split_s
        questions = pd.concat([questions, processed_df_quest])

        s = df.iloc[0, 3]
        split_s = re.split(r"\d+\.", s)
        split_s = [i.strip() for i in split_s if i]
        processed_df_ans["answers"] = split_s
        answers = pd.concat([answers, processed_df_ans])

    questions["role"] = "user"
    questions = questions[["role", "questions"]]
    questions.columns = ["role", "content"]
    questions.index = np.arange(1, len(questions) + 1)

    answers["role"] = "assistant"
    answers = answers[["role", "answers"]]
    answers.columns = ["role", "content"]
    answers.index = np.arange(1, len(answers) + 1)

    final_messages = []
    L = min(len(questions), len(answers))
    for i in range(L):
        final_messages.append(questions.iloc[i, :].to_dict())
        final_messages.append(answers.iloc[i, :].to_dict())

    return final_messages


def extract_data(feed):
    pdf_reader = PdfReader(feed)
    pages = len(pdf_reader.pages)

    text_list = []
    for page in range(pages):
        pdf_page = pdf_reader.pages[page]
        text = pdf_page.extract_text()
        text_list.append(text)

    return text


def internet_search(prompt: str) -> Dict[str, str]:
    content_bodies = []
    list_of_urls = []
    with DDGS() as ddgs:
        i = 0
        for r in ddgs.text(prompt, region="wt-wt", safesearch="Off", timelimit="y"):
            if i <= 5:
                content_bodies.append(r["body"])
                list_of_urls.append(r["href"])
                i += 1
            else:
                break

    return {"context": content_bodies, "urls": list_of_urls}


def read_url(url: str) -> str:
    """
    Reads the contents of a public URL as a string.

    Args:
      url: A public URL.

    Returns:
      A string containing the contents of the URL.
    """

    # Create a request object.
    request = requests.get(url)

    # Check the response status code.
    if request.status_code == 200:
        # The request was successful, so we can read the content.
        return request.content.decode("utf-8")
    else:
        # The request failed, so we raise an exception.
        raise Exception("Unable to read URL.")


def token_size(string):
    tokens = string.split()
    return float(len(tokens))
