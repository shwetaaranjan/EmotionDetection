import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
import json
import numpy as np

from transformers import BertTokenizer, TFBertModel
from sklearn.preprocessing import LabelEncoder

from nltk.tokenize import TreebankWordTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

lemmatizer = WordNetLemmatizer()

# Load intents
with open('C:/Users/ishwe/OneDrive/Desktop/finalyearproject/chatbot/Intents2.json') as f:
    intents = json.load(f)

sentences_p = []
labels = []
words = []
ignore_words = ['?', '!', '.', ',']
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokenizer = TreebankWordTokenizer()
        tokens = tokenizer.tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, intent['tag']))
        sentences_p.append(pattern)
        labels.append(intent['tag'])

# Preprocessing
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

# Lowercase sentences
lower_final_sentences = [s.lower() for s in sentences_p]

# Load Hugging Face BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Convert sentences to BERT token IDs
max_length = 20
bert_input_ids = [tokenizer.encode(s, max_length=max_length, padding='max_length', truncation=True) for s in lower_final_sentences]
train_x = np.array(bert_input_ids)

# Label encode
encoder = LabelEncoder()
train_y = encoder.fit_transform(labels)
train_y = np.array(train_y)

# Build model
model = Sequential()
model.add(Embedding(input_dim=tokenizer.vocab_size, output_dim=64, input_length=max_length, name="embed"))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(len(set(labels)), activation='softmax'))  # use softmax for multi-class classification

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train
model.fit(train_x, train_y, validation_split=0.1, epochs=20, batch_size=5)

# Evaluate
loss, acc = model.evaluate(train_x, train_y)
print("Loss:", loss)
print("Accuracy:", acc)

# Save
model.save('chatbot_bert_lstm.h5')
