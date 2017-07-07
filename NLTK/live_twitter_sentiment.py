from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import sentiment_mod as s

#consumer key, consumer secret, access token, access secret.
ckey="PPpgLL8RL7YeUpC0urA12VdDW"
csecret="bgtSnV6TUdIqAXkqu9ItGPO4aV85puJWNSvp0mLGu9fMMEih1f"
atoken="2566732320-CIpyRS5E0PgWSJbMqwiZGqjgpOwFQ8wwTbtb2yF"
asecret="t5p0TqXrquGrIh0eXAr7X1DiBh3hUeWDehRXx0blRjmOR"

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