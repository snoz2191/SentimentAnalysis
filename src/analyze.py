# -*- coding: utf-8 -*-
__author__ = 'domingo'
import string
import re
from pattern.es import tag
from pattern.vector import SVM, CLASSIFICATION, LINEAR
from pattern.es.wordlist import STOPWORDS as stopwords
import sys
import locale

locale.setlocale(locale.LC_ALL,'es_VE.UTF-8')

#Procedure that initializes the dictionaries needed for the classifier
def get_pol_map():
    #Initialize Dictionaries
    global adj_map
    global adv_map
    global noun_map
    global verb_map
    global abrv_map
    adj_map = {}
    adv_map = {}
    noun_map = {}
    verb_map = {}
    abrv_map = {}
    #Compile Regular Expression
    regex = re.compile(r'[\s]+')
    #Loading Adjectives Dictionary
    d = open("../dictionaries/adj_dict_es.txt")
    for elem in d:
        message = regex.split(elem)
        adj_map[message[0]] = float(message[1])
    d.close()
    #Loading Adverbs Dictionary
    d = open("../dictionaries/adv_dict_es.txt")
    for elem in d:
        message = regex.split(elem)
        adv_map[message[0]] = float(message[1])
    d.close()
    #Loading Nouns Dictionary
    d = open("../dictionaries/noun_dict_es.txt")
    for elem in d:
        message = regex.split(elem)
        noun_map[message[0]] = float(message[1])
    d.close()
    #Loading Verbs Dictionary
    d = open("../dictionaries/verb_dict_es.txt")
    for elem in d:
        message = regex.split(elem)
        verb_map[message[0]] = float(message[1])
    d.close()
    #Loading Slang and Abbreviations Dictionary
    d = open("../dictionaries/abreviaciones_es.txt")
    for elem in d:
        message = regex.split(elem)
        abrv_map[message[0]] = message[1]
    d.close()


#Function that returns a dictionary that contains all the booster words and their booster percentage
def get_booster_map():
    dict = {}
    regex = re.compile(r'[\s]+')
    d = open("../dictionaries/booster_es.txt")
    for elem in d:
        message = regex.split(elem)
        dict[message[0]] = message[1]
    d.close()
    return dict


#Function that returns a dictionary that contains all the negating words and their negating percentage
def get_negator_map():
    dict = {}
    regex = re.compile(r'[\s]+')
    d = open("../dictionaries/negators_es.txt")
    for elem in d:
        message = regex.split(elem)
        dict[message[0]] = message[1]
    d.close()
    return dict


#Function that returns a dictionary that translates the space-separated booster o negating
#phrases to the format of the booster and negator dictionary
def get_mod_map(booster, negator):
    dict = {}
    for elem in booster:
        dict[re.sub(re.escape("_"), " ", elem)] = elem
    for elem in negator:
        dict[re.sub(re.escape("_"), " ", elem)] = elem
    return dict


#Function that takes the train and test files and loads them into memory
def load_data_from_file(train_path, test_path):
    train_set = []
    test_set = []
    train_file = open(train_path)
    for line in train_file:
        tw = line.split('\t')
        if len(tw) != 2:
            continue
        tweet = {}
        tweet['message'] = tw[0]
        tweet['sentiment'] = tw[1]
        train_set.append(tweet)
    test_file = open(test_path)
    for line in test_file:
        tw = line.split('\t')
        if len(tw) != 2:
            continue
        tweet = {}
        tweet['message'] = tw[0]
        tweet['sentiment'] = "unknown"
        test_set.append(tweet)
    return [train_set, test_set]


#Function that preprocesses the given tweet and returns the tweet tagged with his POS tags
def preprocess(tweet):
    message = tweet.decode('utf-8', errors='ignore')

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

    message = message.lower()

    message = re.sub(re.escape('...'),'.' + ' &dots',message)

    #Normalization of punctuation for emphasizing a phrase
    for symbol in string.punctuation:
        message = re.sub(re.escape(symbol)+r'{3,}',' ' + symbol + ' &emphasis',message)

    #Normalization of repeated characters
    for symbol in string.letters:
        message = re.sub(re.escape(symbol)+r'{2,}', symbol+symbol ,message)

    #Replace abbreviations with the full word
    for elem in abrv_map.items():
        message = re.sub(r'(\s+|^)'+re.escape(elem[0])+r'(\s+|$)', r'\g<1>'+elem[1].decode('utf8')+r'\g<2>' , message)

    #Replace booster or negating phrases with said with "_" instead of whitespaces
    for elem in modifiers.items():
        message = re.sub(re.escape(elem[0]), elem[1], message)

    message = re.sub(' +',' ' ,message)
    message = message.strip()

    #Tag each word with the corresponding POS tag
    message = tag(message, tokenize=False)
    return message


#Function that, given a tagged tweet, returns the feature vector that serves as input
#to the SVM classifier. The feature vector is composed as follows:
#   -   Bag of Words Features: For each word given in the train corpus, the position i of the vector is true
#                              if the word corresponding to that position is found on the tweet
#   -   POS Tags: For each tag considered ( NN, VG, CD, JJ, CC, RB ) the value is true if the tag is found
#                 on the tweet
#   -   Presence of Negators
def get_features(tweet):
    global bag_of_words
    if len(bag_of_words) == 0:
        print "NO BAG OF WORDS!!!"
    twords = [word.lower() for word, tag in tweet if word.encode('utf8') not in stopwords and not word.isdigit()]
    ttags = [tag[:2] for word, tag in tweet if word.encode('utf8') not in stopwords and not word.isdigit()]
    feature_set = {}
    for word in bag_of_words:
        feature_set['has_'+word] = (word in twords)
    for tag in ['NN','VG','CD','JJ','CC','RB']:
        feature_set['has_'+tag] = (tag in ttags)
    negators = set(['no', 'nada', 'nadie', 'nunca', 'jamás', 'jamas', 'en_mi_vida','ninguna', 'ninguno'])
    if len(negators.intersection(set(twords))) > 0:
        feature_set['has_negator'] = True
    return feature_set


#Procedure that trains the svm_classifier and produces a model for polarity classification
def train_svm(classifier, tweets):
    global bag_of_words
    print "Training In Progress......"
    bows = {}
    for tweet in tweets:
        for w, t in preprocess(tweet['message']):
            if w.encode('utf8') not in stopwords and not w.isdigit():
                bows[w.lower()] = bows.get(w.lower(),0) + 1
    bag_of_words = [w for w,freq in sorted(bows.items(),key= lambda x: (-x[1], x[0]))[:1000]]
    for tweet in tweets:
        svm_classifier.train(get_features(preprocess(tweet['message'])),type=tweet['sentiment'])


#Function that, given a preprocessed tweet, returns the polarity score of the same based on
#the emoticons present, if any.
def evaluate_emoticons(tweet):
    positive = ['&happy', '&laugh', '&wink', '&heart', '&highfive', '&angel', '&tong']
    negative = ['&sad', '&annoyed', '&seallips', '&devil']
    positive_count = 0
    negative_count = 0
    for t, p in tweet:
        if t in positive:
            print t
            positive_count += 1
        if t in negative:
            print t
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

def get_dict(tag):
    if tag[:2] == "NN":
        return noun_map
    elif tag[:2] == "VB":
        return verb_map
    elif tag[:2] == "RB":
        return adv_map
    else:
        return adj_map

#Function that, given a preprocessed tweet, returns the sentiment orientation of the same based
#on the sentiment of the words used and the presence of boosters, negators and intensifier.
def evaluate_bow_and_features(tweet):
    #global modifiers, booster_map, negator_map
    # tweet = [w.lower() for w,tag in tweet]
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
    for i, (w,tag) in enumerate(tweet):
        bow_map = get_dict(tag)
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
            so = 0.0
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
                                so = so - 5
                            else:
                                so = so + 5
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
                            so = so - 5
                        else:
                            so = so + 5
                    if intensifier and (i-i_int) <= lookup_window:
                        so = so + ( so * intensifier )

            if boost_up:
                so = so*2
            if boost_down:
                so = (so*(1.0))/2

            if so > 0:
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


#3-Layer Hybrid Classifier:
#   1st Layer: Emoticon Presence and Sentiment
#   2st Layer: Sentiment Orientation Calculator. Only if 1st layer evaluates to 0
#   3rd Layer: SVM. Only if all above layers return 0
#Param: Tweet -> String
#Returns: Classification of the tweet in either Positive, Negative or Neutral category
def hybrid_classify(tweet):
    filter_tweet = preprocess(tweet)
    print filter_tweet
    emoticon_result = evaluate_emoticons(filter_tweet)
    if emoticon_result > 0:
        print "EMOTICON RESULT"
        return 'Positive'
    elif emoticon_result < 0:
        print "EMOTICON RESULT"
        return 'Negative'

    bow_result = evaluate_bow_and_features(filter_tweet)
    print bow_result
    if bow_result >= 2:
        print "BOW RESULT"
        return 'Positive'
    elif bow_result <= -2:
        print "BOW RESULT"
        return 'Negative'

    svm_result = svm_classifier.classify(get_features(filter_tweet))
    print "SVM RESULT"
    return svm_result.strip('\n')


#Main Function
def main():
    global modifiers, booster_map, negator_map, bag_of_words
    get_pol_map()
    booster_map = get_booster_map()
    negator_map = get_negator_map()
    modifiers = get_mod_map(booster_map, negator_map)

    bag_of_words = []
    train_data, test_data = load_data_from_file(sys.argv[1], sys.argv[2])
    #output_file = open("output.output", "w")

    global svm_classifier
    svm_classifier = SVM(type=CLASSIFICATION, kernel=LINEAR)
    train_svm(svm_classifier, train_data)

    for n, tweet in enumerate(test_data):
        class1 = hybrid_classify(tweet['message'])
        print class1
        #line = 'sid\tuid\t' + class1 + '\t' + tweet['message']
        #output_file.write(line)
    #output_file.close()


if __name__ == "__main__":
    main()
