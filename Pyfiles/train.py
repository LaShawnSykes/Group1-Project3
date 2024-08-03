from model import create_model
from preprocess import prepare_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def train_model(X, y, vocab_size, max_len, num_classes):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    model = create_model(vocab_size, max_len, num_classes)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint]
    )

    return model, history, X_test, y_test_cat

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X, y, label_encoder, tokenizer = prepare_data('guardian_articles.csv')
    model, history, X_test, y_test = train_model(X, y, vocab_size=10000, max_len=200, num_classes=len(label_encoder.classes_))
    plot_history(history)
