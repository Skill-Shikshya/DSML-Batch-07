import re
import pickle
import numpy as np

from fastapi import FastAPI

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

eng_stopwords= stopwords.words('english')
lemmatizer = WordNetLemmatizer()

lables = {
    "Negative": 0,
    "Positive": 1
}

app = FastAPI()

def text_processor(text):
    text = text.lower()

    # Urls (https ...)
    text = re.sub(r"(http|https)\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,6}(/\S*)?", "", text)
    # HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Puncuation and special Chars
    text = re.sub("[^\w\d\s]", "", text)

    text = text.strip()

    # Stop words
    text = [word for word in text.split() if word not in eng_stopwords]
    text = " ".join(text)
    # Lemmatizer
    text = lemmatizer.lemmatize(text)

    return text

def vectorize(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def pred_pipe(input_text, model, vector_model):
    processed_text = text_processor(input_text)

    embeddings = vectorize(processed_text.split(), vector_model)

    predict = model.predict([embeddings])
    
    return [key for key, value in lables.items() if value == predict[0]]

@app.get("/review/{user_review}")
def main(user_review):
    print(user_review)
    
    with open("Day-59/sentiment_model.pkl", "rb") as f:
        models = pickle.load(f)

    w2v_model = models["embedding"]
    log_reg = models["classefier"]

    sentiment = pred_pipe(user_review, log_reg, w2v_model)
    
    return sentiment