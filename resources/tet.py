import requests

url = "https://api.bing.microsoft.com/v7.0/news/search"
params = {
    'q': 'top stories'  # Query parameter
}
headers = {
    'Ocp-Apim-Subscription-Key': '19ed88ff7f714a6989a1ea817a26d2a3'
}

response = requests.get(url, headers=headers, params=params)
data = response.json()

print(data)


