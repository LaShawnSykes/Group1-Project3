from datetime import datetime, timedelta, date
from guardian_ml_6 import predict_article_type
from fpdf import FPDF
from tensorflow.keras.models import load_model
import pandas as pd 
import tensorflow as tf
import pickle
import openai
import os
import requests
import time
import random
import tensorflow as tf

def load_model_and_dependencies():
    try:
        # Load the pickled results
        open('./models/guardian_article_classifier_final.h5', 'rb')
        
        # Extract the components
        model_path = fold_results['model']  # Assuming this is now a path to the saved model
        tokenizer = fold_results['tokenizer']
        label_encoder = fold_results['label_encoder']
        
        # Load the model with experimental_io_device option
        options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        model = tf.saved_model.load(model_path, options=options)
        
        return model, tokenizer, label_encoder
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # If loading fails, try to load components separately
        try:
            options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
            model = tf.saved_model.load('./models/saved_model', options=options)
            with open('./models/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            with open('./models/label_encoder.pickle', 'rb') as handle:
                label_encoder = pickle.load(handle)
            return model, tokenizer, label_encoder
        except Exception as e:
            print(f"Error loading separate components: {str(e)}")
            raise

def fetch_yesterday_articles():
    api_key = os.getenv('GUARDIAN_API_KEY')
    end_date = date.today()
    start_date = end_date - timedelta(days=1)  # Fetch only yesterday's articles
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    base_url = "https://content.guardianapis.com/search"
    
    sections = ['politics', 'business', 'technology', 'sport', 'culture', 'environment', 'science', 'world']
    all_articles = []

    print(from_date, "  ", to_date)

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
                elif response.status_code == 400:
                    print ("end of list")
                    return None
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
    df.to_csv('.\\resources\\guardian_articles_cleaned.csv', index=False)
    print(f"Total articles fetched: {len(all_articles)}")
    print("DataFrame saved to 'guardian_articles_cleaned.csv'")
    print("Columns in the saved CSV:", df.columns.tolist())
    # Modify guardianapi() to fetch only yesterday's articles
    # yesterday = datetime.date.today() - datetime.timedelta(days=1)
    # ... (implement the rest of the function)
'''
def load_model_and_dependencies():
    model = tf.keras.models.load_model('.\\resources\\guardian_deep_learning_model.h5')
    with open('.\\resources\\guardian_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('.\\resources\\guardian_label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)
    return model, tokenizer, label_encoder
'''
def classify_articles(articles, model, tokenizer, label_encoder):
    classified_articles = []
    for article in articles:
        category = predict_article_type(article['title'], article['body'], model, tokenizer, label_encoder)
        classified_articles.append({**article, 'category': category})
    return classified_articles

def summarize_by_category(classified_articles):
    summaries = {}
    for article in classified_articles:
        category = article['category']
        if category not in summaries:
            summaries[category] = []
        summaries[category].append(f"{article['title']}: {article['body'][:100]}...")
    return summaries

def generate_newspaper_pdf(summaries):
    openai.api_key = 'your-openai-api-key'
    prompt = f"Create a newspaper front page layout with the following summaries:\n\n{summaries}"
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=1000)
    
    layout = response.choices[0].text
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, layout)
    pdf.output("newspaper_front_page.pdf")

def main():
    # load_model_and_dependencies()
    articles = fetch_yesterday_articles()
    model, tokenizer, label_encoder = load_model_and_dependencies()
    classified_articles = classify_articles(articles, model, tokenizer, label_encoder)
    summaries = summarize_by_category(classified_articles)
    generate_newspaper_pdf(summaries)

if __name__ == "__main__":
    main()