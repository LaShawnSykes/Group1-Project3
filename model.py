from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, GlobalMaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_model(vocab_size, max_len, num_classes):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        Conv1D(32, 5, activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling1D(pool_size=4),
        BatchNormalization(),
        LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = create_model(vocab_size=10000, max_len=200, num_classes=5)
    model.summary()
