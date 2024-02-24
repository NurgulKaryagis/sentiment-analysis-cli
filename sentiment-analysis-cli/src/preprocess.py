import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def parse_data(file_path):
    labels = []
    texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            label, text = line.split(' ', 1)
            label = 0 if label == '__label__1' else 1
            labels.append(label)
            texts.append(text.strip())
    return labels, texts

def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    return stemmed_tokens

# TRAIN DATA
raw_train_path = "data/raw/test_sample.ft.txt"
labels_train, texts_train = parse_data(raw_train_path)
texts_train_preprocessed = [" ".join(preprocess_text(text)) for text in texts_train]

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts_train_preprocessed)
sequences_train = tokenizer.texts_to_sequences(texts_train_preprocessed)

# Padding
max_length = max(len(x) for x in sequences_train)  # Use the same max_length for both train and test
sequences_train_padded = pad_sequences(sequences_train, maxlen=max_length)

# TEST DATA
raw_test_path = "data/raw/test_sample.ft.txt"
labels_test, texts_test = parse_data(raw_test_path)
texts_test_preprocessed = [" ".join(preprocess_text(text)) for text in texts_test]

sequences_test = tokenizer.texts_to_sequences(texts_test_preprocessed)
sequences_test_padded = pad_sequences(sequences_test, maxlen=max_length)

# Saving tokenizer and processed data
with open("data/processed/tokenizer.pkl", 'wb') as f:
    pickle.dump(tokenizer, f)

with open("data/processed/train_data.pkl", 'wb') as f:
    pickle.dump((sequences_train_padded, labels_train), f)

with open("data/processed/test_data.pkl", 'wb') as f:
    pickle.dump((sequences_test_padded, labels_test), f)
