import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('GUARDIAN_API_KEY')
BASE_URL = "https://content.guardianapis.com/search"

def fetch_articles(start_date, end_date, section):
    articles = []
    current_date = start_date
    while current_date <= end_date:
        params = {
            'api-key': API_KEY,
            'section': section,
            'from-date': current_date.strftime("%Y-%m-%d"),
            'to-date': (current_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            'show-fields': 'bodyText',
            'page-size': 50
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        articles.extend(data['response']['results'])
        current_date += timedelta(days=1)
        time.sleep(1)  # Rate limiting
    return articles

def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    sections = ['politics', 'business', 'technology', 'sport', 'culture']
    
    all_articles = []
    for section in sections:
        all_articles.extend(fetch_articles(start_date, end_date, section))
    
    df = pd.DataFrame(all_articles)
    df.to_csv('guardian_articles.csv', index=False)
    print(f"Saved {len(df)} articles to guardian_articles.csv")

if __name__ == "__main__":
    main()
