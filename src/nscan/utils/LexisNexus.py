import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# LexisNexis API endpoint and credentials
API_URL = "https://services-api.lexisnexis.com/v1/CompanyAndFinancial"
API_KEY = "your_api_key_here"

# CSV File to store data
CSV_FILE = "financial_news_data_2023.csv"

# Function to fetch financial news for a specific date range
def fetch_financial_news(start_date, end_date):
    headers = {
        "Accept": "application/json;odata.metadata=minimal",
        "Authorization": f"Bearer {API_KEY}",
    }

    # Construct GET request with provided syntax
    params = {
        "$search": "Benzinga.com",  # Adjust the search query as needed
        "$select": "Date,Article_title,Stock_symbol,Url,Publisher,Author,Article",
        "$filter": f"Date ge {start_date}T00:00:00Z and Date le {end_date}T23:59:59Z",
        "$orderby": "relevance",
        "$top": 200,  # Maximum allowed per request
    }

    # Make the API request
    response = requests.get(API_URL, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} for range {start_date} to {end_date}")
        print(response.json())  # Display the error details
        return None

# Function to parse API results
def parse_results(api_results):
    records = []
    for item in api_results.get("value", []):
        record = {
            "Date": item.get("Date"),
            "Article_title": item.get("Article_title"),
            "Stock_symbol": item.get("Stock_symbol"),
            "Url": item.get("Url"),
            "Publisher": item.get("Publisher"),
            "Author": item.get("Author"),
            "Article": item.get("Article"),
        }
        records.append(record)
    return records

# Function to save data to CSV
def save_to_csv(records):
    # Load existing data if CSV exists
    if os.path.exists(CSV_FILE):
        existing_data = pd.read_csv(CSV_FILE)
    else:
        existing_data = pd.DataFrame()

    # Create a new DataFrame with the fetched records
    new_data = pd.DataFrame(records)

    # Append and remove duplicates
    combined_data = pd.concat([existing_data, new_data]).drop_duplicates()

    # Save to CSV
    combined_data.to_csv(CSV_FILE, index=False)
    print(f"Data saved to {CSV_FILE}")

# Main function to fetch and save data for 10-day chunks
def run_task():
    start_date = datetime(2023, 1, 1)  # Starting date
    end_date = datetime(2023, 12, 1)  # Ending date

    current_start_date = start_date

    while current_start_date <= end_date:
        # Calculate the current 10-day range
        current_end_date = min(current_start_date + timedelta(days=9), end_date)

        # Convert to ISO 8601 string format
        start_date_str = current_start_date.strftime("%Y-%m-%d")
        end_date_str = current_end_date.strftime("%Y-%m-%d")

        print(f"Fetching data from {start_date_str} to {end_date_str}...")
        api_results = fetch_financial_news(start_date_str, end_date_str)

        if api_results:
            parsed_data = parse_results(api_results)
            save_to_csv(parsed_data)
        else:
            print(f"No data fetched for range {start_date_str} to {end_date_str}.")

        # Move to the next 10-day range
        current_start_date = current_end_date + timedelta(days=1)

    print("Data collection completed for the specified date range.")

# Run the script
if __name__ == "__main__":
    run_task()
