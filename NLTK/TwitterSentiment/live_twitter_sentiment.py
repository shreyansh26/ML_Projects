from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import sentiment_mod as s

#consumer key, consumer secret, access token, access secret.
ckey="CONSUMER_KEY"
csecret="CONSUMER_KEY_SECRET"
atoken="ACCESS_TOKEN"
asecret="ACCESS_TOKEN_SECRET"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        
        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)

        if confidence*100 >= 70:
            output = open("twitter_out.txt", "a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()
        
        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["fun"])
