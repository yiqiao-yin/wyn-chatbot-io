import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import yfinance as yf
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


def download_stock_data(
    tickers: List[str], start_date: str, end_date: str, w: int
) -> pd.DataFrame:
    """
    Download stock data for given tickers between start_date and end_date.

    Args:
    tickers (List[str]): List of stock ticker symbols.
    start_date (str): Start date for data retrieval in 'YYYY-MM-DD' format.
    end_date (str): End date for data retrieval in 'YYYY-MM-DD' format.
    w (int): Size of the interval that is used to download data

    Returns:
    pd.DataFrame: DataFrame with adjusted close prices for the given tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date, interval=w)
    return data["Adj Close"]


def create_portfolio_and_calculate_returns(
    stock_data: pd.DataFrame, top_n: int
) -> pd.DataFrame:
    """
    Create a portfolio and calculate returns based on the given window size.

    Args:
    stock_data (pd.DataFrame): DataFrame containing stock data.
    window_size (int): Size of the window to calculate returns.

    Returns:
    pd.DataFrame: DataFrame containing calculated returns and portfolio history.
    """
    # Compute returns
    returns_data = stock_data.pct_change()
    returns_data.dropna(inplace=True)

    portfolio_history = []  # To keep track of portfolio changes over time
    portfolio_returns = []  # To store portfolio returns for each period

    # Loop over the data in window_size-day windows
    window_size = 1
    for start in range(0, len(returns_data) - window_size, window_size):
        end = start + window_size
        current_window = returns_data[start:end]
        top_stocks = (
            current_window.mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        next_window = returns_data[end : end + window_size][top_stocks].mean(axis=1)

        portfolio_returns.extend(next_window)
        added_length = len(next_window)
        portfolio_history.extend([top_stocks] * added_length)

    new_returns_data = returns_data.copy()
    new_returns_data = new_returns_data.iloc[0:-window_size, :]
    new_returns_data["benchmark"] = new_returns_data.apply(
        lambda x: x[0:5].mean(), axis=1
    )
    new_returns_data["portfolio_returns"] = portfolio_returns
    new_returns_data["portfolio_history"] = portfolio_history
    new_returns_data["rolling_benchmark"] = (
        new_returns_data["benchmark"] + 1
    ).cumprod()
    new_returns_data["rolling_portfolio_returns"] = (
        new_returns_data["portfolio_returns"] + 1
    ).cumprod()

    return new_returns_data


def run_algotrader() -> Dict[str, Any]:
    """
    Runs an algorithmic trading strategy on a predefined list of stock tickers. It downloads the stock data,
    creates a portfolio, and calculates the returns.

    The function performs its calculations based on historical data starting from a fixed start date until
    the current date. It assumes monthly frequency for the calculations.

    Returns:
    Dict[str, Any]: A dictionary representing the latest row of the calculated returns data.
    """
    # A hardcoded list of ticker symbols to be used in the algorithmic trading strategy
    tickers = [
        "AXP",
        "AMGN",
        "AAPL",
        "BA",
        "CAT",
        "CSCO",
        "CVX",
        "GS",
        "HD",
        "HON",
        "IBM",
        "INTC",
        "JNJ",
        "KO",
        "JPM",
        "MCD",
        "MMM",
        "MRK",
        "MSFT",
        "NKE",
        "PG",
        "TRV",
        "UNH",
        "CRM",
        "VZ",
        "V",
        "WBA",
        "WMT",
        "DIS",
        "DOW",
        "XOM",
        "WFC",
        "MA",
        "COST",
        "AVGO",
        "ADBE",
        "C",
        "NFLX",
        "PYPL",
        "TSLA",
        "NVDA",
    ]

    # There's a bug here: 'tickers' is already a list, so calling 'split' is incorrect
    tickers_list = [
        ticker.strip() for ticker in tickers.split(",")
    ]  # This line should be `tickers_list = tickers`

    # Define the time range for the stock data: from the start of 2020 to the current date
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Download the stock data for the specified tickers and time range with monthly frequency
    stock_data = download_stock_data(
        tickers_list,
        start_date.strftime(
            "%Y-%m-%d"
        ),  # Redundant strftime, since start_date is already a string
        end_date.strftime("%Y-%m-%d"),
        w="1mo",
    )

    # Create a portfolio from the downloaded stock data and calculate returns using a certain logic (not shown)
    returns_data = create_portfolio_and_calculate_returns(stock_data, 5)

    # Return the most recent entry of the return data
    return {"prediction": returns_data.tail(1)}


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

    # Run algotrader
    prediction_ = run_algotrader()

    # Construct the response data including both the question and its corresponding answer
    processed_resp = {"question": question, "answer": answer, "prediction": prediction_}

    # Create HTTP response object with status code, headers, and the JSON payload as body
    http_resp = {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(processed_resp),  # Body is converted to a JSON string
    }

    return http_resp
