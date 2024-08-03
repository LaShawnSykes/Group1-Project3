import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def prepare_data(csv_file, max_words=10000, max_len=200):
    df = pd.read_csv(csv_file)
    df['processed_text'] = df['title'] + ' ' + df['bodyText']
    df['processed_text'] = df['processed_text'].apply(preprocess_text)

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['section'])

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df['processed_text'])
    sequences = tokenizer.texts_to_sequences(df['processed_text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    return padded_sequences, df['label'].values, label_encoder, tokenizer

if __name__ == "__main__":
    X, y, label_encoder, tokenizer = prepare_data('guardian_articles.csv')
    print(f"Prepared {len(X)} articles for training")
