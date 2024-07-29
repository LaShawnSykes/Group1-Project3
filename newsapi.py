import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
import pandas as pd

def newsapi(pages):
    # Your API key
    api_key = os.getenv('NEWS_API_KEY')

    # Set up date range for this month
    end_date = datetime.now()
    start_date = end_date.replace(day=1)  # First day of the current month

    # Convert dates to required format (YYYY-MM-DD)
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")

    # Construct the base URL
    url = "https://newsapi.org/v2/everything"

    # Parameters for the API call
    params = {
        "apiKey": api_key,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,  # maximum allowed per request
        "page": pages,
        "q":"weather"
    }

    all_articles = []
    max_pages = pages  # Maximum number of pages to fetch

    for page in range(1, max_pages + 1):
        params["page"] = page
        
        try:
            response = requests.get(url, params=params)
            print (response.url)
            if response.status_code == 200:
                data = response.json()
                articles = data["articles"]
                
                if not articles:
                    print(f"No more articles found after page {page - 1}")
                    break
                
                all_articles.extend(articles)
                print(f"Fetched page {page}, total articles: {len(all_articles)}")
                
                # Sleep for a short time to avoid hitting rate limits
                time.sleep(0.5)
            else:
                print(f"Error on page {page}: {response.status_code}")
                break
        
        except Exception as e:
            print(f"An error occurred on page {page}: {str(e)}")
            break

        print(f"Total articles fetched: {len(all_articles)}")
        # Create a DataFrame from the fetched articles
        df = pd.DataFrame(all_articles)
        # Display the first few rows and basic info about the DataFrame
        print(df.head())
        print(df.info())

        # Optionally, save the DataFrame to a CSV file
        df.to_csv('.\\resources\\news_articles.csv', index=False)
        print("DataFrame saved to 'news_articles.csv'")


