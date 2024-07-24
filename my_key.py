import os
import requests
from dotenv import load_dotenv


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
                'test_url': 'https://content.guardianapis.com/search?api-key={}'
            },
            'GDELT Project': {
                'env_var': 'GDELT_API_KEY',
                'test_url': 'https://api.gdeltproject.org/api/v2/context/dailycontext?format=html&DATERANGE=1&API_KEY={}'
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
            else api_key:
                response = requests.get(config['test_url'].format(api_key))
                print (config['test_url'], 'code =', {response.status_code})

            assert response.status_code in {200, 429, 404, 401}, f'The key provided failed to authenticate {api_name} API. Status code: {response.status_code} {api_key}'

    except Exception as e:
        print(f'An error occurred: {e}')
        return False
    else:
        print('All keys loaded and authenticated correctly')
        return True

# Usage
if key_check("C:\SRC\.key.env"):
    print("All API keys are valid and working.")
else:
    print("There was an issue with one or more API keys.")