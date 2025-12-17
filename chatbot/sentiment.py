from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load FinBERT once (cached)
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
vader = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    finbert_result = finbert(text)[0]
    vader_result = vader.polarity_scores(text)

    return {
        "FinBERT Label": finbert_result['label'],
        "FinBERT Confidence": round(finbert_result['score'], 3),
        "VADER Positive": vader_result['pos'],
        "VADER Negative": vader_result['neg'],
        "VADER Neutral": vader_result['neu'],
        "VADER Compound": vader_result['compound']
    }
