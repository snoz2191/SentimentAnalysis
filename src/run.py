# -*- coding: utf-8 -*-
import string
from pattern.es import tag
import re
from ngrams import *
from pattern.vector import SVM, CLASSIFICATION, LINEAR
from pattern.en.wordlist import STOPWORDS as stopwords
from operator import itemgetter
from copy import copy
import locale

locale.setlocale(locale.LC_ALL,'es_VE.UTF-8')

def load_data_from_file(train_path, test_path):
    train_set = []
    test_set = []
    train_file = open(train_path)
    for line in train_file:
        tw = line.split('\t')
        if len(tw) != 4:
            continue
        tweet = {}
        tweet['message'] = tw[3]
        tweet['sentiment'] = tw[2]
        train_set.append(tweet)
    test_file = open(test_path)
    for line in test_file:
        tw = line.split('\t')
        if len(tw) != 4:
            continue
        tweet = {}
        tweet['message'] = tw[3]
        tweet['sentiment'] = "unknown"
        test_set.append(tweet)
    return [train_set, test_set]

def preprocess(tweet):
    message = tweet.decode('utf-8', errors='ignore')\
    #remove @ from tweets
    message = re.sub(re.escape('@')+r'(\w+)','&mention \g<1>',message)
    #remove # from tweets
    message = re.sub(re.escape('#')+r'(\w+)','&hastag \g<1>',message)
    #remove urls from tweets
    message = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','&url',message)
    #transform emoticons into emotion to be later analyzed
    emoticons = {':-)': '&happy', ':)': '&happy', ':o)': '&happy', ':]': '&happy', ':3': '&happy', ':c)': '&happy',
                 ':>': '&happy', '=]': '&happy', '8)': '&happy', '=)': '&happy', ':}': '&happy', ':^)': '&happy',
                 ':-))': '&happy', '|;-)': '&happy', ":'-)": '&happy', ":')": '&happy', '\o/': '&happy',
                 '*\\0/*': '&happy', ':-D': '&laugh', ':D': '&laugh', '8-D': '&laugh', '8D': '&laugh', 'x-D': '&laugh',
                 'xD': '&laugh', 'X-D': '&laugh', 'XD': '&laugh', '=-D': '&laugh', '=D': '&laugh', '=-3': '&laugh',
                 '=3': '&laugh', 'B^D': '&laugh', '>:[': '&sad', ':-(': '&sad', ':(': '&sad', ':-c': '&sad',
                 ':c': '&sad', ':-<': '&sad', ':<': '&sad', ':-[': '&sad', ':[': '&sad', ':{': '&sad', ':-||': '&sad',
                 ':@': '&sad', ":'-(": '&sad', ":'(": '&sad', 'D:<': '&sad', 'D:': '&sad', 'D8': '&sad', 'D;': '&sad',
                 'D=': '&sad', 'DX': '&sad', 'v.v': '&sad', "D-':": '&sad', '(>_<)': '&sad', ':|': '&sad',
                 '>:O': '&surprise', ':-O': '&surprise', ':-o': '&surprise', ':O': '&surprise', '°o°': '&surprise',
                 ':O': '&surprise', 'o_O': '&surprise', 'o_0': '&surprise', 'o.O': '&surprise', '8-0': '&surprise',
                 '|-O': '&surprise', ';-)': '&wink', ';)': '&wink', '*-)': '&wink', '*)': '&wink', ';-]': '&wink',
                 ';]': '&wink', ';D': '&wink', ';^)': '&wink', ':-,': '&wink', '>:P': '&tong', ':-P': '&tong',
                 ':P': '&tong', 'X-P': '&tong', 'x-p': '&tong', 'xp': '&tong', 'XP': '&tong', ':-p': '&tong',
                 ':p': '&tong', '=p': '&tong', ':-Þ': '&tong', ':Þ': '&tong', ':-b': '&tong', ':b': '&tong',
                 ':-&': '&tong', ':&': '&tong', '>:\\': '&annoyed', '>:/': '&annoyed', ':-/': '&annoyed',
                 ':-.': '&annoyed', ':/': '&annoyed', ':\\': '&annoyed', '=/': '&annoyed', '=\\': '&annoyed',
                 ':L': '&annoyed', '=L': '&annoyed', ':S': '&annoyed', '>.<': '&annoyed', ':-|': '&annoyed',
                 '<:-|': '&annoyed', ':-X': '&seallips', ':X': '&seallips', ':-#': '&seallips', ':#': '&seallips',
                 'O:-)': '&angel', '0:-3': '&angel', '0:3': '&angel', '0:-)': '&angel', '0:)': '&angel',
                 '0;^)': '&angel', '>:)': '&devil', '>;)': '&devil', '>:-)': '&devil', '}:-)': '&devil',
                 '}:)': '&devil', '3:-)': '&devil', '3:)': '&devil', 'o/\o': '&highfive', '^5': '&highfive',
                 '>_>^': '&highfive', '^<_<': '&highfive', '<3': '&heart'
    }
    for symbol in emoticons:
        message = re.sub(r'('+re.escape(symbol)+r')[^a-z0-9A-Z]',' \g<1> '+emoticons[symbol]+' ',message+' ')
    
    message = re.sub(re.escape('...'),'.' + ' &dots',message)

    #erase repeated characters, replace with max 2 ocurrences
    message = re.sub(r'(.)\1+', r'\1\1', message)

    for symbol in string.punctuation:
        message = re.sub(re.escape(symbol)+r'{3,}',' ' + symbol + ' &emphasis',message)

    for symbol in string.letters:
        message = re.sub(re.escape(symbol)+r'{3,}', symbol ,message)

    message = re.sub(' +',' ' ,message)
    message = message.strip()

    message = tag(message, tokenize=False)

    return message

def evaluate_emoticons(tweet):
    positive = ['&happy', '&laugh', '&wink', '&heart', '&highfive', '&angel', '&tong']
    negative = ['&sad', '&annoyed', '&seallips', '&devil']
    positive_count = 0
    negative_count = 0
    for t, p in tweet:
        if t in positive:
            positive_count += 1
        if t in negative:
            negative_count += 1
    return positive_count - negative_count

def find_word(word, wlist):
    result = None
    matches = [w for w in sorted(wlist) if ( w==word ) or ( w[-1]=='*' and word.startswith(w[:-1]) ) ]
    longest_match = 0
    for match in matches:
        if len(match) >  longest_match:
            longest_match = len(match)
            result = match
    return result

def evaluate_bow_and_features(tweet):
    global bow_map, negator_map, booster_map
    tweet = [w.lower() for w,tag in tweet]
    pos_so = 0.0
    neg_so = 0.0
    intensifier = 0
    negation = False
    i_neg, i_int = 0,0
    lookup_window = 5
    boost_up = False
    boost_down = False
    conj_fol = ['but', 'however', 'nevertheless', 'otherwise', 'yet', 'still', 'nonetheless']
    conj_prev = ['till', 'until', 'despite', 'in spite', 'though', 'although']
    conj_infer = ['therefore', 'furtherore', 'consequently', 'thus', 'as a result', 'subsequently', 'eventuall hence']
    for i,w in enumerate(tweet):
        if conj_fol.count(w) or conj_infer.count(w):
            boost_up = True
            negation = False
            boost_down = False
            intensifier = False
        elif conj_prev.count(w):
            boost_down = True
            boost_up = False
            negation = False
            intensifier = False
        else:
            w = find_word(w, bow_map.keys())
            if w != None and w[0] == '#':
                h_intensifier = 0
                h_negation = False
                h_j_neg, h_j_int = 0,0
                w = w[1:]
                t = segment2(w)[1]
                for j in range(len(t)):
                    w = t[j]
                    if w in bow_map:
                        so = bow_map[w]
                        if h_negation and (j-h_j_neg) <= lookup_window:
                            if so > 0:
                                so = so - 4
                            else:
                                so = so + 4
                        if h_intensifier and (j-h_j_int) <= lookup_window:
                            so = so + ( so * intensifier )
                        if so >0:
                            pos_so += 3*so
                        else:
                            neg_so += 3*so
                        if w in negator_map:
                            negation = True
                            h_j_neg = j
                        if w in booster_map:
                            intensifier = booster_map[w]
                            h_j_int = j

            else:
                if w in bow_map:
                    so = bow_map[w]
                    if negation and (i-i_neg) <= lookup_window:
                        if so > 0:
                            so = so - 4
                        else:
                            so = so + 4
                    if intensifier and (i-i_int) <= lookup_window:
                        so = so + ( so * intensifier )
		    if boost_up:
			so = so*2
		    if boost_down:
			so = (so*(1.0))/2
                    if so >0:
                        pos_so += so
                    else:
                        neg_so += so
            	    if w in negator_map:
                        negation = True
                        i_neg = i
            	    if w in booster_map:
                        intensifier = booster_map[w]
                        i_int = i
    return pos_so+neg_so    

def ml_classify(tweet):
    return svm_classifier.classify(get_features(tweet))

def hybrid_classify(tweet):
    filter_tweet = preprocess(tweet)
    emoticon_result = evaluate_emoticons(filter_tweet)
    if emoticon_result > 0:
        return 'positive'
    elif emoticon_result < 0:
        return 'negative'

    bow_result = evaluate_bow_and_features(filter_tweet)
    if bow_result >= 2:
        return 'positive'
    elif bow_result <= -2:
        return 'negative'
    
    svm_result = ml_classify(filter_tweet)
    #print "\t\t\tSVM RESULT!!"
    return svm_result

def get_bow_map():
    dictionary={}

    fp = open("Data/EmotionLookupTable.txt");
    for line in fp:
        line = line.split('\t')
        if len(line) >= 2:
            word = line[0].strip()
            word = word.decode()
            so = line[1].strip()
            dictionary[word] = float(so)
    fp.close()

    fp = open("Data/EmoticonLookupTable.txt");
    for line in fp:
        line = line.split('\t')
        if len(line) >= 2:
            word = line[0].strip()
            word = word.decode(errors='replace')
            so = line[1].strip()
            dictionary[word] = float(so)
    fp.close()
            
    return dictionary

def get_booster_map():
    booster_list={}
    fp = open("Data/BoosterWordList.txt");
    for line in fp:
        line = line.split('\t')
        if len(line) >= 2:
            word = line[0].strip()
            word = word.decode()
            so = line[1].strip()
            booster_list[word] = float(so)
    fp.close()
    return booster_list

def get_negator_map():
    negation_list=[]
    fp = open("Data/NegatingWordList.txt");
    for line in fp:
        word = line.strip()
        word = word.decode()
        negation_list.append(word)
    fp.close()
    return negation_list

def get_features(tweet):
    global bag_of_words
    if len(bag_of_words) == 0:
        print "NO BAG OF WORDS!!!"
    twords = [word.lower() for word, tag in tweet if word not in stopwords and not word.isdigit()]
    ttags = [tag[:2] for word, tag in tweet if word not in stopwords and not word.isdigit()]
    feature_set = {}
    for word in bag_of_words:
        feature_set['has_'+word] = (word in twords)
    for tag in ['NN','VG','CD','JJ','CC','RB']:
        feature_set['has_'+tag] = (tag in ttags)
    negators = set(['not', 'none', 'nobody', 'never', 'nothing', 'lack', 't','n\'t','dont', 'no'])
    if len(negators.intersection(set(twords))) > 0:
        feature_set['has_negator'] = True
    return feature_set

def train_svm(classifier, tweets):
    global bag_of_words, svm_classifier
    print "Training"
    bows = {}
    ws = []
    for tweet in tweets:
        for w, t in preprocess(tweet['message']):
            if w not in stopwords and not w.isdigit():
                ws.append(w.lower())
        for w in ws:
            bows[w] = bows.get(w,0) + 1
    bag_of_words = [w for w,freq in sorted(bows.items(),key=itemgetter(1),reverse=True)[:1000]]
    for tweet in tweets:
        svm_classifier.train(get_features(preprocess(tweet['message'])),type=tweet['sentiment'])

bow_map = get_bow_map()
booster_map = get_booster_map()
negator_map = get_negator_map()

bag_of_words = []

train_data, test_data = load_data_from_file("Data/tweeti-b.dist.data", "Data/twitter-test.tsv")

output_file = open("output.output", "w")

svm_classifier = SVM(type=CLASSIFICATION, kernel=LINEAR)
train_svm(svm_classifier, train_data)

for n, tweet in enumerate(test_data):
    class1 = hybrid_classify(tweet['message'])
    print class1
    line = 'sid\tuid\t' + class1 + '\t' + tweet['message']
    output_file.write(line)
output_file.close()
