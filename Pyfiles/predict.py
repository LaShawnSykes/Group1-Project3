from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from preprocess import preprocess_text

def load_components():
    model = load_model('best_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)
    return model, tokenizer, label_encoder

def predict_category(title, body, model, tokenizer, label_encoder, max_len=200):
    text = f"{title} {body}"
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded)
    predicted_class = label_encoder.classes_[prediction.argmax()]
    return predicted_class

if __name__ == "__main__":
    model, tokenizer, label_encoder = load_components()
    
    title = "New breakthrough in renewable energy"
    body = "Scientists have discovered a revolutionary method to harness solar power, potentially solving the world's energy crisis."
    
    category = predict_category(title, body, model, tokenizer, label_encoder)
    print(f"Predicted category: {category}")
