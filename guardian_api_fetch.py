import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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

def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    freq = FreqDist(words)
    
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if i in sentence_scores:
                    sentence_scores[i] += freq[word]
                else:
                    sentence_scores[i] = freq[word]
    
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = ' '.join([sentences[i] for i in sorted(top_sentences)])
    
    return summary

def get_top_snippets(articles, n=3):
    def get_article_date(article):
        return article.get('webPublicationDate', '')
    
    sorted_articles = sorted(articles, key=get_article_date, reverse=True)
    snippets = []
    for article in sorted_articles[:n]:
        title = article.get('webTitle', 'No title')
        body = article.get('fields', {}).get('bodyText', 'No description')
        snippet = f"{title}: {body[:100]}..."
        snippets.append(snippet)
    return snippets

def get_news_summary(topic, language='en', sort='newest', limit=10):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    sections = ['politics', 'business', 'technology', 'sport', 'culture']
    all_articles = []
    
    for section in sections:
        all_articles.extend(fetch_articles(start_date, end_date, section))
    
    # Filter articles based on the topic
    filtered_articles = [article for article in all_articles if topic.lower() in article['webTitle'].lower()]
    
    # Sort articles
    if sort == 'newest':
        filtered_articles.sort(key=lambda x: x['webPublicationDate'], reverse=True)
    elif sort == 'oldest':
        filtered_articles.sort(key=lambda x: x['webPublicationDate'])
    
    # Limit the number of articles
    filtered_articles = filtered_articles[:limit]
    
    # Generate summary
    all_content = " ".join([article['fields']['bodyText'] for article in filtered_articles])
    summary = summarize_text(all_content, num_sentences=3)
    
    # Get top snippets
    top_snippets = get_top_snippets(filtered_articles)
    
    return filtered_articles, summary, top_snippets

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
