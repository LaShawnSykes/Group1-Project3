import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import re
import random
import pickle
import ast
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
from textblob import TextBlob
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Concatenate, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras.layers import BatchNormalization, SpatialDropout1D

from keras_tuner import RandomSearch

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
    df.to_csv('.\\resources\\guardian_articles_cleaned.csv', index=False)
    print(f"Total articles fetched: {len(all_articles)}")
    print("DataFrame saved to 'guardian_articles_cleaned.csv'")
    print("Columns in the saved CSV:", df.columns.tolist())
    

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

def extract_features(text):
    blob = TextBlob(text)
    return {
        'text_length': len(text),
        'word_count': len(text.split()),
        'sentiment': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
    }

def safe_json_loads(x):
    try:
        return json.loads(x.replace("'", '"'))
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(x)
        except:
            print(f"Failed to parse: {x}")
            return {}

def calculate_category_scores(y_true, y_pred, label_encoder):
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    target_names = [label_encoder.classes_[i] for i in unique_classes if i < len(label_encoder.classes_)]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    category_scores = {category: report[category]['f1-score'] for category in target_names}
    return category_scores

def lr_schedule(epoch):
    if epoch < 10:
        return 0.001
    elif epoch < 20:
        return 0.0005
    else:
        return 0.0001

class AdaptiveClassWeightCallback(Callback):
    def __init__(self, class_weight_dict, patience=3):
        super(AdaptiveClassWeightCallback, self).__init__()
        self.class_weight_dict = class_weight_dict
        self.patience = patience
        self.wait = 0
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.class_weight_dict = {k: v * 1.1 for k, v in self.class_weight_dict.items()}
                print(f"\nAdjusting class weights: {self.class_weight_dict}")
                self.wait = 0

def build_model_tunable(hp, num_classes):
    max_words = 10000
    max_len = 200
    num_classes = hp.Choice('num_classes', values=[8])  # Assuming 8 categories
    
    text_input = Input(shape=(max_len,))
    embedding = Embedding(max_words, hp.Int('embedding_dim', min_value=100, max_value=200, step=50), input_length=max_len)(text_input)
    embedding = SpatialDropout1D(hp.Float('spatial_dropout', min_value=0.1, max_value=0.3, step=0.1))(embedding)
    
    # CNN layers
    conv1 = Conv1D(hp.Int('conv_filters', min_value=64, max_value=256, step=64), 
                   hp.Int('conv_kernel_size', min_value=3, max_value=7, step=2), 
                   activation='relu', kernel_regularizer=l2(hp.Float('conv_l2', min_value=1e-5, max_value=1e-3, sampling='log')))(embedding)
    conv1 = BatchNormalization()(conv1)
    pool1 = GlobalMaxPooling1D()(conv1)
    
    # LSTM layers
    lstm = Bidirectional(LSTM(hp.Int('lstm_units', min_value=32, max_value=256, step=32), 
                              return_sequences=True, 
                              kernel_regularizer=l2(hp.Float('lstm_l2', min_value=1e-5, max_value=1e-3, sampling='log'))))(embedding)
    lstm = BatchNormalization()(lstm)
    lstm = Bidirectional(LSTM(hp.Int('lstm_units_2', min_value=32, max_value=256, step=32), 
                              kernel_regularizer=l2(hp.Float('lstm_l2_2', min_value=1e-5, max_value=1e-3, sampling='log'))))(lstm)
    lstm = BatchNormalization()(lstm)
    
    # Combine CNN and LSTM
    concat_cnn_lstm = Concatenate()([pool1, lstm])
    
    additional_input = Input(shape=(4,))  # 4 additional features
    
    concat = Concatenate()([concat_cnn_lstm, additional_input])
    dense = Dense(hp.Int('dense_units', min_value=128, max_value=1024, step=128), 
                  activation='relu', 
                  kernel_regularizer=l2(hp.Float('dense_l2', min_value=1e-5, max_value=1e-3, sampling='log')))(concat)
    dense = BatchNormalization()(dense)
    dropout = Dropout(hp.Float('dropout', min_value=0.3, max_value=0.6, step=0.1))(dense)
    dense = Dense(hp.Int('dense_units_2', min_value=64, max_value=512, step=64), 
                  activation='relu', 
                  kernel_regularizer=l2(hp.Float('dense_l2_2', min_value=1e-5, max_value=1e-3, sampling='log')))(dropout)
    dense = BatchNormalization()(dense)
    dropout = Dropout(hp.Float('dropout_2', min_value=0.3, max_value=0.6, step=0.1))(dense)
    output = Dense(num_classes, activation='softmax')(dropout)

    model = Model(inputs=[text_input, additional_input], outputs=output)

    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))
    else:
        optimizer = RMSprop(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def build_model(max_words, max_len, additional_features_shape, num_classes):
    text_input = Input(shape=(max_len,))
    embedding = Embedding(max_words, 200, input_length=max_len)(text_input)
    
    # CNN layers
    conv1 = Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.01))(embedding)
    pool1 = GlobalMaxPooling1D()(conv1)
    
    # LSTM layers
    lstm = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)))(embedding)
    lstm = Bidirectional(LSTM(64, kernel_regularizer=l2(0.01)))(lstm)
    
    # Combine CNN and LSTM
    concat_cnn_lstm = Concatenate()([pool1, lstm])
    
    additional_input = Input(shape=(additional_features_shape,))
    
    concat = Concatenate()([concat_cnn_lstm, additional_input])
    dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(concat)
    dropout = Dropout(0.5)(dense)
    dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dropout)
    dropout = Dropout(0.5)(dense)
    output = Dense(num_classes, activation='softmax')(dropout)

    model = Model(inputs=[text_input, additional_input], outputs=output)

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=0.5, random_state=42)  
    print("Columns in the CSV file:", df.columns.tolist())

    title_column = 'webTitle' if 'webTitle' in df.columns else df.columns[0]
    
    if 'fields' in df.columns:
        print("Found 'fields' column. Extracting body text from it.")
        df['body_text'] = df['fields'].apply(lambda x: safe_json_loads(x).get('bodyText', ''))
        df['text'] = df[title_column] + ' ' + df['body_text'].fillna('')
    else:
        print("No 'fields' column found. Using only the title.")
        df['text'] = df[title_column]

    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Additional features
    additional_features = pd.DataFrame(df['text'].apply(extract_features).tolist())
    
    # Prepare the data
    X_text = df['processed_text'].values
    X_additional = additional_features.values
    y = df['category'].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weight_dict = dict(enumerate(class_weights))

    # Implement Stratified K-fold cross-validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    max_words = 10000
    max_len = 200

    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_text, y_encoded)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")

        # Split the data
        X_text_train, X_text_val = X_text[train_idx], X_text[val_idx]
        X_add_train, X_add_val = X_additional[train_idx], X_additional[val_idx]
        y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]

        # Tokenize the text
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(X_text_train)
        X_train_seq = tokenizer.texts_to_sequences(X_text_train)
        X_val_seq = tokenizer.texts_to_sequences(X_text_val)
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
        X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)

        # Hyperparameter tuning
        tuner = RandomSearch(
        lambda hp: build_model_tunable(hp, num_classes),
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory=f'tuning_results_fold_{fold}',
        project_name='guardian_article_classification'
        )

        tuner.search([X_train_pad, X_add_train], y_train,
                     validation_data=([X_val_pad, X_add_val], y_val),
                     epochs=20, batch_size=32)

        best_model = tuner.get_best_models(num_models=1)[0]

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(f'.\\resources\\model_fold_{fold+1}.h5', 
                                           monitor='val_loss', save_best_only=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        adaptive_class_weight = AdaptiveClassWeightCallback(class_weight_dict)

        # Train the best model
        history = best_model.fit(
            [X_train_pad, X_add_train], y_train,
            validation_data=([X_val_pad, X_add_val], y_val),
            epochs=30,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, model_checkpoint, lr_scheduler, adaptive_class_weight]
        )

        # Evaluate the model
        val_loss, val_accuracy = best_model.evaluate([X_val_pad, X_add_val], y_val)
        print(f"Validation accuracy for fold {fold + 1}: {val_accuracy:.4f}")

        # Calculate and print category scores
        y_pred = best_model.predict([X_val_pad, X_add_val])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_val_classes = np.argmax(y_val, axis=1)
        category_scores = calculate_category_scores(y_val_classes, y_pred_classes, label_encoder)
        
        fold_scores.append({
            'val_accuracy': val_accuracy,
            'category_scores': category_scores
        })

    # Print average scores across all folds
    print("\nAverage scores across all folds:")
    avg_accuracy = np.mean([score['val_accuracy'] for score in fold_scores])
    print(f"Average validation accuracy: {avg_accuracy:.4f}")

    avg_category_scores = {}
    for category in label_encoder.classes_:
        avg_score = np.mean([score['category_scores'][category] for score in fold_scores])
        avg_category_scores[category] = avg_score
        print(f"Average {category} F1-score: {avg_score:.4f}")

    # Train final model on all data
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_text)
    X_seq = tokenizer.texts_to_sequences(X_text)
    X_pad = pad_sequences(X_seq, maxlen=max_len)
    
    final_tuner = RandomSearch(
        build_model_tunable,
        objective='val_accuracy',
        max_trials=20,
        executions_per_trial=2,
        directory='final_model_tuning',
        project_name='guardian_article_classification_final'
    )

    final_tuner.search([X_pad, X_additional], y_categorical,
                       validation_split=0.2,
                       epochs=50,
                       batch_size=32,
                       callbacks=[early_stopping, lr_scheduler, adaptive_class_weight])

    best_model = final_tuner.get_best_models(num_models=1)[0]
    
    # Final training on all data
    best_model.fit(
        [X_pad, X_additional], y_categorical,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, lr_scheduler, adaptive_class_weight]
    )

    # Save the final model and necessary objects
    best_model.save('.\\resources\\guardian_deep_learning_model.h5')
    with open('.\\resources\\guardian_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('.\\resources\\guardian_label_encoder.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return best_model, tokenizer, label_encoder, avg_category_scores

def predict_article_type(title, body, model, tokenizer, label_encoder):
    text = f"{title} {body}"
    processed_text = preprocess_text(text)
    additional_features = extract_features(text)
    
    # Tokenize and pad the text
    text_seq = tokenizer.texts_to_sequences([processed_text])
    text_pad = pad_sequences(text_seq, maxlen=2)  # Make sure this matches the max_len in train_model
    
    # Prepare additional features
    add_features = np.array([list(additional_features.values())])
    
    # Make prediction
    prediction = model.predict([text_pad, add_features])
    predicted_class_index = np.argmax(prediction)
    if predicted_class_index < len(label_encoder.classes_):
        predicted_class = label_encoder.classes_[predicted_class_index]
    else:
        predicted_class = "Unknown"
    
    return predicted_class # may need []

if __name__ == "__main__":
    if os.getenv("GUARDIAN_API_KEY"):
        print("Guardian API key is valid and working.")
        # guardianapi()
        model, tokenizer, label_encoder, category_scores = train_model('.\\resources\\guardian_articles_cleaned.csv')

        print("\nFinal Category F1-scores:")
        for category, score in category_scores.items():
            print(f"{category}: {score:.4f}")

        # Test the model
        test_articles = [
            ("Climate Change Impact", "Global temperatures rise, causing extreme weather events worldwide."),
            ("Tech Company Layoffs", "Major tech firms announce significant job cuts due to economic downturn."),
            ("Sports Championship", "Local team wins national title in a thrilling overtime victory."),
            ("Political Scandal", "High-ranking official resigns amid corruption allegations.")
        ]

        for title, body in test_articles:
            prediction = predict_article_type(title, body, model, tokenizer, label_encoder)
            print(f"\nArticle: {title}")
            print(f"Predicted category: {prediction}")
    else:
        print("There was an issue with the Guardian API key.")