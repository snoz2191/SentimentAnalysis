# -*- coding: utf-8 -*-
__author__ = 'domingo'
import string
import re
from pattern.es import tag
from pattern.vector import SVM, CLASSIFICATION, LINEAR
from pattern.es.wordlist import STOPWORDS as stopwords
import sys
import locale
import os

locale.setlocale(locale.LC_ALL,'es_VE.UTF-8')

#Procedure that initializes the dictionary needed for the classifier
def get_pol_map():
	#Initialize Dictionary
	global bow_map
	global abrv_map
	bow_map = {}
	abrv_map = {}
	#Compile Regular Expression
	regex = re.compile(r'[\s]+')
	#Loading Adjectives Dictionary
	file = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'dictionaries', 'sentiment_map_es.txt')
	d = open(file)
	for elem in d:
		message = regex.split(elem)
		bow_map[message[0]] = float(message[1])
	d.close()
	#Loading Slang and Abbreviations Dictionary
	file = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'dictionaries', 'abreviaciones_es.txt')
	d = open(file)
	for elem in d:
		message = regex.split(elem)
		abrv_map[message[0]] = message[1]
	d.close()


#Function that returns a dictionary that contains all the booster words and their booster percentage
def get_booster_map():
	dict = {}
	regex = re.compile(r'[\s]+')
	file = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'dictionaries', 'booster_es.txt')
	d = open(file)
	for elem in d:
		message = regex.split(elem)
		dict[message[0]] = float(message[1])
	d.close()
	return dict


#Function that returns a dictionary that contains all the negating words and their negating percentage
def get_negator_map():
	dict = {}
	regex = re.compile(r'[\s]+')
	file = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'dictionaries', 'negators_es.txt')
	d = open(file)
	for elem in d:
		message = regex.split(elem)
		dict[message[0]] = float(message[1])
	d.close()
	return dict


#Function that returns a dictionary that translates the hyphen-separated booster or negating
#phrases to a space-separated format
def get_mod_map(booster, negator):
	dict = {}
	for elem in booster:
		dict[re.sub(re.escape("_"), " ", elem)] = elem
	for elem in negator:
		dict[re.sub(re.escape("_"), " ", elem)] = elem
	return dict


#Function that takes the train and test files and loads them into memory
def load_data_from_file(train_path, test_path, crowd_path):
	train_set = []
	test_set = []
	train_file = open(train_path)
	for line in train_file:
		tw = line.split('\t')
		if len(tw) != 2:
			continue
		tweet = {'message': tw[0].strip(), 'sentiment': tw[1].strip()}
		train_set.append(tweet)
	test_file = open(test_path)
	for line in test_file:
		tw = line.split('\t')
		if len(tw) != 2:
			continue
		tweet = {'message': tw[0].strip(), 'sentiment': tw[1].strip()}
		test_set.append(tweet)
	crowd_file = open(crowd_path)
	crowd_set = []
	for line in crowd_file:
		tw = line.split('\t')
		if len(tw) != 2:
			continue
		tweet = {'message': tw[0].strip(), 'sentiment': tw[1].strip()}
		crowd_set.append(tweet)
	return [train_set, test_set,crowd_set]


#Function that preprocesses the given tweet and returns the tweet tagged with his POS tags
def preprocess(tweet):
	message = tweet.decode('utf-8', errors='ignore')

	#remove @ from tweets
	message = re.sub(re.escape('@')+r'(\w+)','&mention \g<1>',message)

	#remove # from tweets
	message = re.sub(re.escape('#')+r'(\w+)','&hashtag \g<1>',message)

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

	#Separation of punctuations from words
	for symbol in ["¡","!",",",".","¿","?"]:
		message = re.sub(r'([0-9A-Za-z]+|^)' + re.escape(symbol) + r'($|\s)', r'\g<1> '+ symbol +r'\g<2>', message)
		message = re.sub(r'(\s|^)' + re.escape(symbol) + r'($|[0-9A-Za-z]+)', r'\g<1> '+ symbol +r' \g<2>', message)
		message = re.sub(r'([0-9A-Za-z]+)' + re.escape(symbol) + r'(0-9A-Za-z]+)', r'\g<1> '+ symbol +r' \g<2>', message)

	#Normalization of repeated characters
	for symbol in string.letters:
		message = re.sub(re.escape(symbol)+r'{2,}', symbol+symbol ,message)

	#Replace abbreviations with the full word
	for elem in abrv_map.items():
		message = re.sub(r'(\s+|^)'+re.escape(elem[0])+r'(\s+|$)', r'\g<1>'+elem[1].decode('utf8')+r'\g<2>' , message)

	#Replace booster or negating phrases with said with "_" instead of whitespaces
	for elem in modifiers.items():
		message = re.sub(r'(\s|^)' + re.escape(elem[0]) + r'(\s|$)' , r'\g<1>'+elem[1]+r'\g<2>', message)

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
	negators = set(
		['no', 'nada', 'nadie', 'nunca', 'jamás', 'jamas', 'ni', 'en_mi_vida', 'ninguna', 'ninguno', 'tampoco',
		 'con_todo_y_con_eso', 'por_el_contrario', 'por_contra', 'antes_bien',
		 'a_pesar_de_todo', 'en_absoluto'])
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
			#print t
			positive_count += 1
		if t in negative:
			#print t
			negative_count += 1
	return positive_count - negative_count


def find_word(word, wlist):
	result = None
	matches = [w for w in sorted(wlist) if ( w == word.encode('utf8') ) or ( w[-1]=='*' and word.startswith(w[:-1]) ) ]
	longest_match = 0
	for match in matches:
		if len(match) >  longest_match:
			longest_match = len(match)
			result = match
	return result


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
	boost_inverted = False
	conj_inverted = ['sino']
	conj_fol = ['aún_asi', 'aun_asi', 'hasta_que', 'no_obstante']
	conj_prev = ['a_pesar_de', 'aunque', 'excepto', 'salvo', 'pero', 'sin_embargo']
	conj_infer = ['por_lo_tanto', 'además', 'ademas', 'en_consecuencia', 'por_tanto', 'como_resultado', 'posteriormente', 'por_eso', 'hasta']
	for i, (w,tag) in enumerate(tweet):
		if conj_fol.count(w.encode('utf8')) or conj_infer.count(w.encode('utf8')):
			boost_up = True
			boost_inverted = False
			negation = False
			boost_down = False
			intensifier = False
		elif conj_prev.count(w.encode('utf8')):
			boost_down = True
			boost_inverted = False
			boost_up = False
			negation = False
			intensifier = False
		elif conj_inverted.count(w.encode('utf8')):
			boost_inverted = True
			boost_down = False
			boost_up = False
			negation = False
			intensifier = False
		else:
			so = 0.0 #Added By DOMINGO
			word = find_word(w, bow_map.keys())
			if word != None and word[0] == '#':
				w = word
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
				so = so*2 #Tweak with the boosting factor!!!!
			if boost_down:
				so = (so*(1.0))/2
			if boost_inverted:
				so = (so*1.0)/(-2)

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
	global EMO, BOW, AI
	Positive = False
	Neutral = False
	Negative = False
	filter_tweet = preprocess(tweet)
	#print filter_tweet
	emoticon_result = evaluate_emoticons(filter_tweet)
	if emoticon_result > 0:
		#print "EMOTICON RESULT"
		Positive = True
	elif emoticon_result < 0:
		#print "EMOTICON RESULT"
		Negative += True
	else:
		Neutral += True

	bow_result = evaluate_bow_and_features(filter_tweet)
	#print bow_result
	if bow_result >= 2:
		#print "BOW RESULT"
		if Positive:
			return 'Positive'
		Positive = True
	elif bow_result <= -2:
		#print "BOW RESULT"
		if Negative:
			return 'Negative'
		Negative = True
	else:
		#if Neutral:
		#	return 'Neutral'
		Neutral = True


	svm_result = svm_classifier.classify(get_features(filter_tweet))
	svm_result = svm_result.strip('\n')
	if svm_result == 'Positive':
		return 'Positive'
	elif svm_result == 'Negative':
		return 'Negative'
	else:
		return 'Neutral'


#Main Function
def main():
	global modifiers, booster_map, negator_map, bag_of_words
	get_pol_map()
	booster_map = get_booster_map()
	negator_map = get_negator_map()
	modifiers = get_mod_map(booster_map, negator_map)

	bag_of_words = []
	train_data, test_data, crowd_data = load_data_from_file(sys.argv[1], sys.argv[2], sys.argv[3])

	global svm_classifier
	svm_classifier = SVM(type=CLASSIFICATION, kernel=LINEAR)
	train_svm(svm_classifier, train_data)
	print "Training Completed......"
	hits = 0.0
	misses = 0.0
	counter = 0.0
	confussion = {}
	global EMO, BOW, AI
	EMO = 0
	BOW = 0
	AI = 0
	for n, tweet in enumerate(test_data):
		class1 = hybrid_classify(tweet['message'])
		counter += 1
		if ( class1 == tweet['sentiment'] ):
			hits += 1
		else:
			misses += 1
		confussion[(class1,tweet['sentiment'])] = confussion.get((class1,tweet['sentiment']),0) + 1
	Accuracy = 	hits/(hits+misses)
	Recall = (hits+misses)/counter
	F1 = (2*Accuracy*Recall)/(Accuracy+Recall)
	print ""
	print "TASS Test Results......"
	print "Accuracy: ",str(Accuracy)
	print "Recall: ",str(Recall)
	print "F1-Score: ", str(F1)
	print "Layer Summary:"
	print "Emoticon Layer: ", str(EMO)
	print "BOW Layer: ", str(BOW)
	print "SVM Layer: ", str(AI)
	print "Confussion Matrix:"
	for elem in confussion.items():
		print elem[0], "\t", str(elem[1])

	hits = 0.0
	misses = 0.0
	counter = 0.0
	EMO = 0
	BOW = 0
	AI = 0
	confussion = {}
	for n, tweet in enumerate(crowd_data):
		class1 = hybrid_classify(tweet['message'])
		counter += 1
		if class1 == tweet['sentiment']:
			hits += 1
		else:
			misses += 1
		confussion[(class1,tweet['sentiment'])] = confussion.get((class1,tweet['sentiment']),0) + 1
	Accuracy = 	hits/(hits+misses)
	Recall = (hits+misses)/counter
	F1 = (2*Accuracy*Recall)/(Accuracy+Recall)
	print ""
	print "Crowd Test Results......"
	print "Accuracy: ",str(Accuracy)
	print "Recall: ",str(Recall)
	print "F1-Score: ", str(F1)
	print "Layer Summary:"
	print "Emoticon Layer: ", str(EMO)
	print "BOW Layer: ", str(BOW)
	print "SVM Layer: ", str(AI)
	print "Confussion Matrix:"
	for elem in confussion.items():
		print elem[0], "\t", str(elem[1])

if __name__ == "__main__":
	main()
