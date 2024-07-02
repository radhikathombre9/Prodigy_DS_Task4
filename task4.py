# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:45:14 2024

@author: Radhika
"""

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"D:\Nikhil Analytics\Prodigy Infotech\demonetization-tweets_data.csv",encoding='ISO-8859-1')
df.head()

texts = df['text']
texts

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

df['sentiment'] = texts.apply(get_sentiment)

def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_class'] = df['sentiment'].apply(classify_sentiment)

print(df[['text', 'sentiment', 'sentiment_class']])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

sentiment_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
df['sentiment_label'] = df['sentiment_class'].map(sentiment_map)

X = df['text']
y = df['sentiment_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

new_texts = ["This is an amazing product!", "I am very disappointed with the service.", "It's okay, not great but not bad either."]
new_texts_tfidf = vectorizer.transform(new_texts)
predictions = model.predict(new_texts_tfidf)
predicted_sentiments = [list(sentiment_map.keys())[list(sentiment_map.values()).index(label)] for label in predictions]

print('Predicted Sentiments:')
for text, sentiment in zip(new_texts, predicted_sentiments):
    print(f'Text: {text} - Sentiment: {sentiment}')


# =============================================================================
#  accuracy = 0.86 
# =============================================================================

plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment_class', data=df, palette='viridis')
plt.title('Count Plot of Sentiment Classes')
plt.xlabel('Sentiment Class')
plt.ylabel('Count')
plt.show()

































