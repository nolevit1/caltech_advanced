#!/usr/bin/env python3

"""
A simple Python script using Flask to serve a webpage that shows the latest Bitcoin price.
This version includes error handling for request and JSON parsing.
"""

from flask import Flask, render_template_string
import requests

app = Flask(__name__)

@app.route('/')
def index():
    # Try to fetch the latest Bitcoin price data from CoinDesk
    try:
        response = requests.get("https://api.coindesk.com/v1/bpi/currentprice.json", timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error fetching data: {e}"

    try:
        data = response.json()
        # Extract the price in USD
        price = data['bpi']['USD']['rate']
        # Create an HTML page
        html = f"""
        <!DOCTYPE html>
        <html lang=\"en\">
        <head>
            <meta charset=\"UTF-8\">
            <title>Bitcoin Price</title>
        </head>
        <body>
            <h1>Current Bitcoin Price:</h1>
            <p><strong>{price} USD</strong></p>
        </body>
        </html>
        """
        return render_template_string(html)
    except (KeyError, ValueError) as e:
        return f"Error parsing JSON data: {e}"

if __name__ == "__main__":
    # Run the Flask development server
    app.run(debug=True, port=5000)
