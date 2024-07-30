import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import re
import random
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib
import json
from textblob import TextBlob

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

load_dotenv('C:\\SRC\\.key.env')

def guardianapi():
    api_key = os.getenv('GUARDIAN_API_KEY')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1200)  # Fetch up to 1200 days of data
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    base_url = "https://content.guardianapis.com/search"
    
    sections = ['politics', 'business', 'technology', 'sport', 'culture', 'environment', 'science', 'world']
    all_articles = []

    def fetch_with_retry(params, max_retries=10, initial_wait=10):
        for attempt in range(max_retries):
            try:
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = min(initial_wait * (2 ** attempt), 60) + random.uniform(0, 1)
                    print(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds.")
                    time.sleep(wait_time)
                else:
                    response.raise_for_status()
            except requests.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("Max retries reached. Skipping this request.")
                    return None
                wait_time = min(initial_wait * (2 ** attempt), 60) + random.uniform(0, 1)
                print(f"Waiting for {wait_time:.2f} seconds before retrying.")
                time.sleep(wait_time)

    for section in sections:
        page = 1
        section_articles = []
        while len(section_articles) < 2000:
            params = {
                "api-key": api_key,
                "from-date": from_date,
                "to-date": to_date,
                "order-by": "relevance",
                "show-fields": "all",
                "page-size": 50,
                "section": section,
                "page": page
            }

            data = fetch_with_retry(params)
            if data is None:
                print(f"Failed to fetch articles from {section}, page {page}")
                break

            articles = data['response']['results']
            if not articles:
                break
            for article in articles:
                article['category'] = section
            section_articles.extend(articles)
            print(f"Fetched page {page} from {section}, total articles: {len(section_articles)}")
            page += 1

            # Random delay between requests
            time.sleep(random.uniform(1, 3))

        all_articles.extend(section_articles[:2000])  # Ensure we only take 2000 articles per section
        print(f"Completed fetching articles for {section}. Total articles: {len(all_articles)}")

    df = pd.DataFrame(all_articles)
    df.to_csv('.\\resources\\guardian_articles_balanced.csv', index=False)
    print(f"Total articles fetched: {len(all_articles)}")
    print("DataFrame saved to 'guardian_articles_balanced.csv'")
    print("Columns in the saved CSV:", df.columns.tolist())

def extract_body_text(fields_str):
    try:
        fields = json.loads(fields_str.replace("'", '"'))
        return fields.get('bodyText', '')
    except:
        return ''

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = clean_text(text)
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

def extract_features(text):
    blob = TextBlob(text)
    return {
        'text_length': len(text),
        'word_count': len(text.split()),
        'sentiment': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'noun_phrase_count': len(blob.noun_phrases)
    }

def train_model(csv_path):
    df = pd.read_csv(csv_path)
    print("Columns in the CSV file:", df.columns.tolist())

    title_column = 'webTitle' if 'webTitle' in df.columns else df.columns[0]
    
    if 'fields' in df.columns:
        print("Found 'fields' column. Extracting body text from it.")
        df['body_text'] = df['fields'].apply(extract_body_text)
        df['text'] = df[title_column] + ' ' + df['body_text'].fillna('')
    else:
        print("No 'fields' column found. Using only the title.")
        df['text'] = df[title_column]

    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), min_df=2, max_df=0.95)
    tfidf_features = vectorizer.fit_transform(df['processed_text'])
    
    # Additional features
    additional_features = pd.DataFrame(df['text'].apply(extract_features).tolist())
    
    # Combine all features
    X = np.hstack((tfidf_features.toarray(), additional_features))
    y = df['category']

    # Use stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # XGBoost classifier with balanced class weights
    class_weights = dict(zip(y.unique(), [1/y.value_counts()[c] for c in y.unique()]))
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        scale_pos_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )

    # Perform cross-validation
    scores = cross_val_score(classifier, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
    print(f"Cross-validation F1 scores: {scores}")
    print(f"Mean F1 score: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")

    # Train on the full dataset
    classifier.fit(X, y)

    joblib.dump(vectorizer, '.\\resources\\guardian_tfidf_vectorizer_improved.joblib')
    joblib.dump(classifier, '.\\resources\\guardian_classifier_improved.joblib')
    print("Model and vectorizer saved for future use.")

    return vectorizer, classifier

def predict_article_type(title, body, vectorizer, classifier):
    text = f"{title} {body}"
    processed_text = preprocess_text(text)
    tfidf_features = vectorizer.transform([processed_text])
    additional_features = pd.DataFrame([extract_features(text)])
    X = np.hstack((tfidf_features.toarray(), additional_features))
    prediction = classifier.predict(X)[0]
    return prediction

if __name__ == "__main__":
    if os.getenv("GUARDIAN_API_KEY"):
        print("Guardian API key is valid and working.")
        guardianapi()
        vectorizer, classifier = train_model('.\\resources\\guardian_articles_balanced.csv')

        # Test the model
        test_articles = [
            ("Climate Change Impact", "Global temperatures rise, causing extreme weather events worldwide."),
            ("Tech Company Layoffs", "Major tech firms announce significant job cuts due to economic downturn."),
            ("Sports Championship", "Local team wins national title in a thrilling overtime victory."),
            ("Political Scandal", "High-ranking official resigns amid corruption allegations.")
        ]

        for title, body in test_articles:
            prediction = predict_article_type(title, body, vectorizer, classifier)
            print(f"\nArticle: {title}")
            print(f"Predicted category: {prediction}")
    else:
        print("There was an issue with the Guardian API key.")