# -*- coding: utf-8 -*-
__author__ = 'domingo'
import time
import sys
import locale
from xml.etree import ElementTree as ET
from twython import Twython, TwythonError, TwythonAuthError, TwythonRateLimitError

locale.setlocale(locale.LC_ALL,'es_VE.UTF-8')

def get_tweet(tweetid, counter, pol):
    global last_update
    # print tweetid.text + " " + pol
    tdiff = time.time() - last_update
    if tdiff < QUERY_PER_SEC:
            time.sleep(QUERY_PER_SEC-tdiff)
    last_update = time.time()
    try:
        status = twitter.show_status(id= tweetid.text)
        counter += 1
        out = status['text'].replace('\n', ' ')
        file.write(out.encode('utf8') + "\t" + pol + "\n")
    except TwythonError as e:
        print e
    return counter

def main():
    global twitter, file, data, QUERY_PER_SEC, last_update
    Nones = 0
    Posi = 0
    Neu = 0
    Neg = 0
    data = []
    ckey = 'p8577Kdv57LcFVbcbDrBPN1ii'
    csecret = 'ZHs7stlQ13MS94uVmkKx70h0hGeW6RsysdoTmmSCk9OqfAKbE7'
    atoken = '119568905-baBtnSZrZvKGoVZYWCZhU8McfMa64oMp6QIiHOqL'
    asecret = 'zZmITh9UODDiHvOYA8bnffhLW1GZzWy6u2HT958QzC7dH'
    twitter = Twython(ckey,csecret,atoken,asecret)
    rootElement = ET.parse(sys.argv[1]).getroot()
    file = open(sys.argv[2], 'w+')
    QUERY_PER_SEC = (15*60)/180.0
    last_update = 0

    for subelement in rootElement:
        #print "Ciclo"
        tweetid = subelement.find('tweetid')
        sentiments = subelement.find('sentiments')
        for polarity in sentiments.findall('polarity'):
            value = polarity.find('value')
            if value.text in ["NONE","P","P+","N","N+","NEU"]: break
        #print tweetid.text + " " + value.text
        if value.text == "NONE":
            Nones = get_tweet(tweetid, Nones, "Neutral")
        elif value.text in ["P","P+"]:
            Posi = get_tweet(tweetid, Posi, "Positive")
        elif value.text in ["N","N+"]:
            Neg = get_tweet(tweetid, Neg, "Negative")
        else:
            Neu = get_tweet(tweetid, Neu, "Neutral")

    file.close()
    print "Summary..."
    print "Positives: ",str(Posi)
    print "Negatives: ",str(Neg)
    print "Neutrals: ",str(Neu)
    print "Nones: ",str(Nones)
    print "Total: ",str(Posi+Neu+Neg+Nones)

if __name__ == "__main__":
    main()