import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def train_model():
    with open('data/processed/train_data.pkl', 'rb') as file:
        train_obj = pickle.load(file)

    with open('data/processed/test_data.pkl', 'rb') as file:
        test_obj = pickle.load(file)

    with open('data/processed/tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

    sequences_train, labels_train = train_obj
    sequences_test, labels_test = test_obj

    sequences_train = np.array(sequences_train)
    labels_train = np.array(labels_train)
    sequences_test = np.array(sequences_test)
    labels_test = np.array(labels_test)

    # Calculating max_length based on sequence lengths
    sequence_lengths = [len(x) for x in sequences_train]  # sequences_train should be used
    max_length = np.percentile(sequence_lengths, 90)  # Find length below 90% of sequences
    max_length = int(max_length)

    vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
    embed_dim = 64  # embedding dimension
    lstm_units = 32  # lstm units

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_length),
        LSTM(lstm_units),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(sequences_train, labels_train, epochs=10, validation_data=(sequences_test, labels_test))
    model.summary()
    model.save('models/model.h5')
    
    return model

def save_model(model, model_path='models/model.h5'):
    model.save(model_path)
    
    