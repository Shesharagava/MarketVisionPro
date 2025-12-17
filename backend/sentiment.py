from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load FinBERT once (cached)
# Using a pipeline is easier for inference
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", top_k=None)
vader = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    # Returns list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
    raw_results = finbert(text)[0] 
    
    # Convert to simple dict { "positive": 0.9, ... }
    scores = {item['label']: round(item['score'], 3) for item in raw_results}
    
    # Find dominant
    dominant_label = max(scores, key=scores.get)
    dominant_score = scores[dominant_label]

    vader_result = vader.polarity_scores(text)

    return {
        "FinBERT Label": dominant_label,
        "FinBERT Confidence": dominant_score,
        "FinBERT Scores": scores,
        "VADER Positive": vader_result['pos'],
        "VADER Negative": vader_result['neg'],
        "VADER Neutral": vader_result['neu'],
        "VADER Compound": vader_result['compound']
    }
