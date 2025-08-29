import nltk
nltk.download('punkt_tab')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import random
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import os

# Correct base path for all files

file_path = os.path.join('C:/', 'Users', 'ishwe', 'OneDrive', 'Desktop', 'finalyearproject', 'Flask', 'Intents2.json')
base_path = os.path.dirname(file_path)

print("file_path:", file_path)  # Debug print

with open(file_path, 'r', encoding='utf-8') as f:
    intents = json.load(f)



words = []
classes = []
documents = []
sentences_p = []
ignore_words = ['?', '!', '.', ',']

# Tokenizing and preparing data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        sentences_p.append(pattern)
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save the pickled words and classes using absolute paths
pickle.dump(words, open(os.path.join(base_path, 'words.pkl'), 'wb'))
pickle.dump(classes, open(os.path.join(base_path, 'classes.pkl'), 'wb'))

# Load pickled files and intents again using absolute paths
intents = json.loads(open(file_path, 'r', encoding='utf-8').read())
words = pickle.load(open(os.path.join(base_path, 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(base_path, 'classes.pkl'), 'rb'))

# Load Sentence Transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def chat(user_input, model):
    user_input = user_input.lower()
    embedding = embedder.encode([user_input])  # shape: (1, embedding_dim)
    embedding = np.expand_dims(embedding, axis=1)
    print("embedding.shape:", embedding.shape)


    results = model.predict(embedding)  # Make sure model input shape matches embedding shape
    result_index = np.argmax(results)
    tag = classes[result_index]
    
    for t in intents['intents']:
        if t['tag'] == tag:
            responses = t['responses']
            return random.choice(responses)
    
    return "Sorry, I didn't understand that. Can you please rephrase?"

# Example usage:
# model = tf.keras.models.load_model(os.path.join(base_path, 'your_trained_model.h5'))
# print(chat("Hello!", model))
