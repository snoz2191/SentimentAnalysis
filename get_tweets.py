from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

ckey = 'p8577Kdv57LcFVbcbDrBPN1ii'
csecret = 'ZHs7stlQ13MS94uVmkKx70h0hGeW6RsysdoTmmSCk9OqfAKbE7'
atoken = '119568905-baBtnSZrZvKGoVZYWCZhU8McfMa64oMp6QIiHOqL'
asecret = 'zZmITh9UODDiHvOYA8bnffhLW1GZzWy6u2HT958QzC7dH'

class listener(StreamListener):

    def on_data(self, data):
        print data
        return True

    def on_error(self, status):
        print status

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = Stream(auth, listener())
twitterStream.filter(track=["car"])
