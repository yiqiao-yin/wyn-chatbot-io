import json
import os
import openai

openai.api_key = os.environ['OPENAI_API_KEY']

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

def lambda_handler(event, context):
    # Grab Data
    question = event['queryStringParameters']["query"]
    user_key = event['queryStringParameters']["key"]

    if user_key == "123":
        answer = call_chatgpt(question)
    else:
        answer = "Please enter the correct key!"

    processed_resp = {
        "question": question,
        "answer": answer
    }
    
    # Create return body
    http_resp = {}
    http_resp['statusCode'] = 200
    http_resp['headers'] = {}
    http_resp['headers']['Content-Type'] = 'application/json'
    http_resp['body'] = json.dumps(processed_resp)

    return http_resp