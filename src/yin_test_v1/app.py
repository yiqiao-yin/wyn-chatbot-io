import json
import os
from typing import Any, Dict

from openai import OpenAI

key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=key)


def call_chatgpt(question: str) -> str:
    """
    Calls the GPT model using the provided client to generate a response to a given question.

    The function takes a question as input and interacts with an OpenAI GPT model through the client's chat
    completion API. If there is any exception during the API call, it catches the exception and provides a
    default error response.

    Parameters:
    question (str): The user's question that needs to be sent to the GPT model.

    Returns:
    str: The generated response from the GPT model. In case of an error, a default error message is returned.
    """
    try:
        # Generate a response using the GPT model specified, with a fixed system role message followed by the user's question.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )
        # Extract the content of the message received from the GPT model.
        output = response.choices[0].message.content
    except Exception as e:
        # Print the exception and return a default error message if any exception occurs.
        print(e)
        output = "Sorry, I couldn't get an answer for that."

    return output


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda function that processes an incoming HTTP request with a question and a user key. If the key is correct,
    it utilizes a function `call_chatgpt` to get an answer for the question; otherwise, it responds with an error
    message.

    The function expects an 'event' object typically provided by AWS API Gateway with the following structure:
    {
        "queryStringParameters": {
            "query": "<User's question>",
            "key": "<User access key>"
        }
    }

    Parameters:
    event (Dict[str, Any]): The event data containing the queryStringParameters with a "query" and a "key".
    context (Any): Provides information about the invocation, function, and execution environment.

    Returns:
    Dict[str, Any]: A dictionary object with statusCode, headers, and body, suitable for returning from an
                    AWS Lambda function behind API Gateway.
    """
    # Grab Data
    question = event["queryStringParameters"]["query"]
    user_key = event["queryStringParameters"]["key"]

    # Validate the provided user key and obtain the response based on the validity of the key
    if user_key == "123":
        answer = call_chatgpt(question)
    else:
        answer = "Please enter the correct key!"

    # Construct the response data including both the question and its corresponding answer
    processed_resp = {"question": question, "answer": answer}

    # Create HTTP response object with status code, headers, and the JSON payload as body
    http_resp = {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(processed_resp),  # Body is converted to a JSON string
    }

    return http_resp
