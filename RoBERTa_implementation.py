import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Connecting with the pretrained model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def get_sentiment(text):
     # Tokenize the input text
    encoded_text = tokenizer(text, return_tensors='pt')
    # Get model outputs
    output = model(**encoded_text)
    # Get scores (logits)
    scores = output[0][0].detach().numpy()
    # Apply softmax to get probabilities
    scores = softmax(scores)
    # Define the labels corresponding to each class
    labels = ['negative', 'neutral', 'positive']
    # Get the index of the class with the highest score (probability)
    predicted_class = np.argmax(scores)
    # Get the confidence of the predicted class
    confidence = scores[predicted_class]
    
    return labels[predicted_class], confidence

# Test the function with an example sentence
text = "I love to eat noodeles and play football!"
sentiment, confidence = get_sentiment(text)

print(f"Sentiment: {sentiment}, Confidence: {confidence:.4f}")



# read the csv file
data_frames = pd.read_csv('amazon_reviews.csv')

data_frames = data_frames[['overall', 'reviewText']]

reviewList = data_frames['reviewText'].tolist()
score = data_frames['overall'].tolist()

print(reviewList[45])
print(score[45])

scoreList = []
for i in score:
    if i == 'Positive':
        scoreList.append(0)
    if i == 'Neutral':
        scoreList.append(1)
    if i == 'Negative':
        scoreList.append(2)


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


result = {}
indices = []

# Storing the accuracy value in 'result' dictionary
# {roberta_neg:value, roberta_neu: value, roberta_pos: value}
# value -> [0, 1]

for i in range(len(reviewList)):
    try:
        result[i] = polarity_scores_roberta(reviewList[i])
    except RuntimeError:
        indices.append(i)
        print(f'Broke for index {i}')

roberta_score = []

for i in result:
    value = max(result[i], key=result[i].get)
    if value == 'roberta_pos':
        roberta_score.append(2)
    if value == 'roberta_neu':
        roberta_score.append(1)
    if value == 'roberta_neg':
        roberta_score.append(0)

roberta_score = np.array(roberta_score)
scoreList = np.array(scoreList)
# print(roberta_score.shape)
# print(scoreList.shape)

for i in indices:
    np.delete(scoreList, i)

print(roberta_score.shape)
print(scoreList.shape)

print(accuracy_score(scoreList, roberta_score))
print(confusion_matrix(scoreList, roberta_score))
