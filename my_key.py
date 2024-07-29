import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
import pandas as pd
import random
'''
key_check 
newsapi



'''

def key_check(key_path=None):
    try:
        load_dotenv(key_path, override=True)
        
        api_configs = {
            'NewsAPI': {
                'env_var': 'NEWS_API_KEY',
                'test_url': 'https://newsapi.org/v2/top-headlines?country=us&apiKey={}'
            },
            'New York Times': {
                'env_var': 'NYT_API_KEY',
                'test_url': 'https://api.nytimes.com/svc/mostpopular/v2/viewed/1.json?api-key={}'
            },
            'The Guardian': {
                'env_var': 'GUARDIAN_API_KEY',
                'test_url': xxg7q0
            },
            'GDELT Project': {
                'env_var': 'GDELT_API_KEY',
                'test_url': 'http://api.gdeltproject.org/api/v1/search_ftxtsearch/search_ftxtsearch?query=heat wave&format=json&maxrecords=250&timespan=1d'
            },
            'Currents API': {
                'env_var': 'CURRENTS_API_KEY',
                'test_url': 'https://api.currentsapi.services/v1/latest-news?apiKey={}'
            },
            'Event Registry': {
                'env_var': 'EVENT_REGISTRY_API_KEY',
                'test_url': 'https://eventregistry.org/api/v1/article/getArticles?apiKey={}'
            },
            'MediaStack API': {
                'env_var': 'MEDIASTACK_API_KEY',
                'test_url': 'http://api.mediastack.com/v1/news?access_key={}'
            },
        }

        for api_name, config in api_configs.items():
            api_key = os.getenv(config['env_var'])
            assert api_key is not None, f'{config["env_var"]} not found in .env file'

            if 'headers' in config:
                headers = {k: v.format(api_key) for k, v in config['headers'].items()}
                response = requests.get(config['test_url'], headers=headers)
                print(f"bing {config['test_url']}, headers={headers}, code = {response.status_code}")
            elif api_key == 'NA':
                response = requests.get(config['test_url'])
                print (config['test_url'], 'code =', {response.status_code})
            else:
                response = requests.get(config['test_url'].format(api_key))
                print (config['test_url'], 'code =', {response.status_code})
            assert response.status_code in {200, 503, 404, 401}, f'The key provided failed to authenticate {api_name} API. Status code: {response.status_code} {api_key}'

    except Exception as e:
        print(f'An error occurred: {e}')
        return False
    else:
        print('All keys loaded and authenticated correctly')
        return True

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

    for page in range(202, max_pages + 1):
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
        df.to_csv('.\\resources\\news_articles_1.csv', index=False)
        print("DataFrame saved to 'news_articles.csv'")

def make_request(url, params, retries=5, backoff_factor=5):
    for i in range(retries):
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                wait_time = (backoff_factor ** i) + random.random()
                print(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds.")
                time.sleep(wait_time)
            else:
                print(f"Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Request failed: {str(e)}")
    return None

def nytapi():
    api_key = os.getenv('NYT_API_KEY')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    begin_date = start_date.strftime("%Y%m%d")
    end_date = end_date.strftime("%Y%m%d")

    base_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

    params = {
        "api-key": api_key,
        "begin_date": begin_date,
        "end_date": end_date,
        "sort": "newest",
        "page": 0
    }

    all_articles = []
    total_pages = 1
    current_page = 0

    while current_page < total_pages:
        response = make_request(base_url, params)
        
        if response:
            data = response.json()
            
            if current_page == 0:
                total_pages = min(data['response']['meta']['hits'] // 10 + 1, 2000)
                print(f"Total pages to fetch: {total_pages}")
            
            articles = data['response']['docs']
            all_articles.extend(articles)
            print(f"Fetched page {current_page + 1}, total articles: {len(all_articles)}")
            
            current_page += 1
            params['page'] = current_page
            
            time.sleep(6)  # Base wait time between requests
        else:
            print(f"Failed to fetch page {current_page + 1}")
            break

    print(f"Total articles fetched: {len(all_articles)}")

    df = pd.DataFrame(all_articles)

    print(df.head())
    print(df.info())

    df.to_csv('.\\resources\\nyt_articles.csv', index=False)
    print("DataFrame saved to 'nyt_articles.csv'")

    return None

if key_check("C:\SRC\.key.env"):
    print("All API keys are valid and working.")
else:
    print("There was an issue with one or more API keys.")
# newsapi(5)
nytapi()