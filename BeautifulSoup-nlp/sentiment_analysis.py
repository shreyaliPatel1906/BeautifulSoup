from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment_textblob(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return 'positive 😊'
    elif sentiment < 0:
        return 'negative 😡'
    else:
        return 'neutral 😑'

def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    compound = sentiment['compound']
    if compound >= 0.05:
        return 'positive 😊'
    elif compound <= -0.05:
        return 'negative 😡'
    else:
        return 'neutral 😑'

def analyze_input():
    text = input('Enter your text: ')
    print(f"TextBlob Sentiment: {analyze_sentiment_textblob(text)}")
    print(f"VADER Sentiment: {analyze_sentiment_vader(text)}")

analyze_input()
