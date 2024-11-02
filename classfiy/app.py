# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from app_store_scraper import AppStore
from google_play_scraper import reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from langdetect import detect
import nltk
import re
import pickle

# Initialize Flask app
app = Flask(__name__)

nltk.data.path.append("G:/NBE/classify/")  # Add the correct path
nltk.download('punkt', download_dir="G:/NBE/classify/")
nltk.download('stopwords', download_dir="G:/NBE/classify/")
nltk.download('wordnet', download_dir="G:/NBE/classify/")
nltk.download('punkt_tab', download_dir="G:/NBE/classify/")
# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('arabic'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        sentence = request.form['sentence']
        preprocessed_sentence = preprocess_text(sentence)
        sentence_vec = vectorizer.transform([preprocessed_sentence])
        prediction = model.predict(sentence_vec)[0]

        if prediction == 1:
            result = "إيجابي 😊"
        elif prediction == -1:
            result = "سلبي 😢"
        else:
            result = "محايد 😐"

    return render_template('index.html', result=result)

# Sentence to be classified


if __name__ == '__main__':
    app.run(debug=True)
