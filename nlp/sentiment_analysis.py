from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict


def _initialize_dict():
    return {
        "Strongly Positive":0,
        "Positive" : 0,
        "Negative" : 0,
        "Strongly Negative" : 0,
        "Neutral" : 0
    }

def _sentiment_score(sentence, strongly_positive_thres, strongly_negative_thres):

    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    # negative = sentiment_dict['neg']
    # neutral = sentiment_dict['neu']
    # positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if compound >= strongly_positive_thres :
        overall_sentiment = "Strongly Positive"
    elif compound >= 0.05 :
        overall_sentiment = "Positive"
    elif compound <= strongly_negative_thres :
        overall_sentiment = "Strongly Negative"
    elif compound <= - 0.05 :
        overall_sentiment = "Negative"        
    else :
        overall_sentiment = "Neutral"
  
    return overall_sentiment

def sentiment_list_score(sentence_list:List, strongly_positive_thres, strongly_negative_thres):
    sentence_analysis_result = _initialize_dict()
    for sentence in sentence_list:
        feeling = _sentiment_score(sentence, strongly_positive_thres, strongly_negative_thres)
        sentence_analysis_result[feeling]+=1
    return sentence_analysis_result