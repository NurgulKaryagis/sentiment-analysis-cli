import argparse
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# train_model and save_model functions are defined in model.py
from models import model

# Setting up the argument parser
parser = argparse.ArgumentParser(description="Sentiment Analysis CLI Tool")
parser.add_argument('--train', action='store_true', help="Train the sentiment analysis model.")
parser.add_argument('--save', type=str, help="Save the trained model to the specified path. Use with --train.")
parser.add_argument('--predict', type=str, help="Predict sentiment of the provided text.")
parser.add_argument('--model', type=str, default='models/model.h5', help="Path to the trained model file.")
parser.add_argument('--tokenizer', type=str, default='data/processed/tokenizer.pkl', help="Path to the tokenizer file.")
args = parser.parse_args()

def predict_sentiment(text, model_path, tokenizer_path):
    """Predicts the sentiment of the provided text using a trained model."""
    # Load the tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load the model
    model = load_model(model_path)
    
    # Process the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=155)  # Adjust `maxlen` based on your training data
    
    # Make a prediction
    prediction = model.predict(padded_sequence)
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return sentiment

if __name__ == '__main__':
    if args.train:
        # Train the model
        model = model.train_model()
        print("Model training completed.")
        if args.save:
            # Save the model to the path specified by --save
            model.save_model(model, args.save)
            print(f"Model has been saved to {args.save}.")
    elif args.predict:
        # Predict sentiment if the --predict argument is used
        sentiment = predict_sentiment(args.predict, args.model, args.tokenizer)
        print(f"Predicted sentiment: {sentiment}")
    else:
        print("No action specified. Use --train to train the model and optionally --save to specify the save path, or --predict to predict sentiment of text.")
