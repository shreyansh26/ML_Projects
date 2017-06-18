# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 07:21:23 2017

@author: Shreyansh Singh
"""

import tweepy
from textblob import TextBlob
import pandas as pd

def get_tweet_sentiment(tweet):
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'	

consumer_key = 'npzAVJDYceV6ZSqs19oBFbCM3'
consumer_secret = '2WvyqnrRYd3gIcbxzNnqpoDbYlq5KQy6URa4r6pDjTQ5A0xVAt'

access_token = '2566732320-AFIiSNjFpCMHOxzoquKZYW5iBC1bSQvy4pxYdU5'
access_token_secret = 'Uq2HDzo74GwM6NBx8rib37GWYRDsDcRxuW0dMzuDxJPZV'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

tweet_list = []
analysis_list = []

for tweet in public_tweets:
	#print(tweet.text.encode("utf-8", errors='ignore'))
	tweet_list.append(tweet.text.encode("utf-8", errors='ignore'))
	analysis_list.append(get_tweet_sentiment(tweet.text))

submission = pd.DataFrame({
        "Tweet": tweet_list,
        "Label": analysis_list
    })

submission.to_csv('./submission.csv', index=False)
