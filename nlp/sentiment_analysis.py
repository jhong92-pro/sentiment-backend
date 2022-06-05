from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_score(sentence, strongly_positive_thres, strongly_negative_thres):

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
  
    return compound, overall_sentiment

