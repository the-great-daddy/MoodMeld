# TEXT BASED SENTIMENT ANALYSIS USING NAIVE BAYES CLASSIFIER

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("stopwords")

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_frames = pd.read_csv("twitter_comments.csv")

data_frames = data_frames[["Text", "Ratings"]]

reviewList = data_frames["Text"].tolist()
score = data_frames["Ratings"].tolist()

reviewList = reviewList[:59880]
score = score[:59880]

scoreList = []
for i in score:
    if i == "Positive":
        scoreList.append(2)
    if i == "Neutral":
        scoreList.append(1)
    if i == "Negative":
        scoreList.append(0)


def tokenizer(text):
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove special characters
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra spaces
    text = re.sub(" +", " ", text)
    # Trim leading and trailing whitespaces
    text = text.strip()
    # Tokenize the text into words
    tokens = word_tokenize(text)
    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()
    # Remove stopwords and stem the tokens
    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stopwords.words("english")
    ]

    return tokens


reviewList = np.array(reviewList)
scoreList = np.array(scoreList)

# Create TF-IDF again: stopwords-> we filter out common words (I,my, the, and...)
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=tokenizer,
    stop_words=stopwords.words("english"),
    lowercase=True,
    token_pattern=None,
)

# Builds a TF-IDF matrix for the sentences
try:
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviewList)
    print(tfidf_matrix.shape)
except ValueError:
    print("Broke")

tfidf_matrix.shape

x_train, x_test, y_train, y_test = train_test_split(
    tfidf_matrix, scoreList, test_size=0.2, random_state=True
)

model = MultinomialNB().fit(x_train, y_train)
predicted = model.predict(x_test)

print(accuracy_score(y_test, predicted))
print(confusion_matrix(y_test, predicted))
