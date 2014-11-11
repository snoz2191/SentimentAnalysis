# -*- coding: utf-8 -*-
#Polarity Classifier using an Horizontal 3-Layer Architecture and a modifies POS Tags Features
#Latest Stable Version: 1.0 - 10/11/2014
__author__ = 'domingo'
import string
import re
from pattern.es import tag
from pattern.vector import SVM, CLASSIFICATION, LINEAR
from pattern.es.wordlist import STOPWORDS as stopwords
import locale
import os
import select
import socket
import sys

locale.setlocale(locale.LC_ALL,'es_VE.UTF-8')

#Get_Pol_Map()
#Procedure that initializes the dictionary needed for the classifier
def get_pol_map():
	#Initialize Dictionaries
	global bow_map
	global abrv_map
	bow_map = {}
	abrv_map = {}
	#Compile Regular Expression
	regex = re.compile(r'[\s]+')
	#Loading Main Dictionary
	file = os.path.join(os.path.dirname(__file__), os.pardir, 'dictionaries', 'sentiment_map_es.txt')
	d = open(file)
	for elem in d:
		message = regex.split(elem)
		bow_map[message[0]] = float(message[1])
	d.close()
	#Loading Slang and Abbreviations Dictionary
	file = os.path.join(os.path.dirname(__file__), os.pardir, 'dictionaries', 'abreviaciones_es.txt')
	d = open(file)
	for elem in d:
		message = regex.split(elem)
		abrv_map[message[0]] = message[1]
	d.close()


#Get_Booster_Map:
#Function that returns a dictionary that contains all the booster words and their booster percentage
def get_booster_map():
	#Initialize Dictionary
	dict = {}
	#Compiling regular expression
	regex = re.compile(r'[\s]+')
	#Loading Booster Words/Phrases Dicitonary
	file = os.path.join(os.path.dirname(__file__), os.pardir, 'dictionaries', 'booster_es.txt')
	d = open(file)
	for elem in d:
		message = regex.split(elem)
		dict[message[0]] = float(message[1])
	d.close()
	return dict

#Get_Negator_Map:
#Function that returns a dictionary that contains all the negating words and their negating percentage
def get_negator_map():
	#Initialize Dictionary
	dict = {}
	#Compiling regular expression
	regex = re.compile(r'[\s]+')
	#Loading Negating Words/Phrases Dicitonary
	file = os.path.join(os.path.dirname(__file__), os.pardir, 'dictionaries', 'negators_es.txt')
	d = open(file)
	for elem in d:
		message = regex.split(elem)
		dict[message[0]] = float(message[1])
	d.close()
	return dict


#Get_Mood_Map:
#Arguments:
#	- booster -> booster words dictionary
#	- negator -> negating words dictionary
#Function that returns a dictionary that translates the hyphen-separated booster or negating
#phrases to a space-separated format
def get_mod_map(booster, negator):
	dict = {}
	for elem in booster:
		dict[re.sub(re.escape("_"), " ", elem)] = elem
	for elem in negator:
		dict[re.sub(re.escape("_"), " ", elem)] = elem
	return dict


#Load_data_from_file:
#Arguments:
#	- train_path -> address of the training file
#Function that takes the train and test files and loads them into memory
def load_data_from_file(train_path):
	train_set = []
	train_file = open(train_path)
	for line in train_file:
		tw = line.split('\t')
		if len(tw) != 2:
			continue
		tweet = {'message': tw[0].strip(), 'sentiment': tw[1].strip()}
		train_set.append(tweet)
	return train_set


#Preprocess:
#Arguments:
#	- tweet -> Tweet to be preprocessed
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

#Get_Features:
#Function that, given a tagged tweet, returns the feature vector that serves as input
#to the SVM classifier. The feature vector is composed as follows:
#   -   Bag of Words Features: For each word given in the train corpus, the position i of the vector is true
#                              if the word corresponding to that position is found on the tweet
#   -   POS Tags: For each tag considered ( NN, VG, CD, JJ, CC, RB ) the number of words that are positive, negative and neutral
#   -   Presence of Negators
#	-	Presence of Positive Emoticons
#	-	Presence of Negative Emoticons
def get_features(tweet):
	global bag_of_words, negator_map, booster_map
	if len(bag_of_words) == 0:
		print "NO BAG OF WORDS!!!"
	twords = [word.lower() for word, tag in tweet if word.encode('utf8') not in stopwords and not word.isdigit()]
	ttags = [tag[:2] for word, tag in tweet if word.encode('utf8') not in stopwords and not word.isdigit()]
	feature_set = {}
	#Bag of words features
	for word in bag_of_words:
		feature_set['has_'+word] = (word in twords)
	#POS Tags features
	for i, w in enumerate(twords):
		if ttags[i] in ['NN','VG','CD','JJ','CC','RB']:
			pol = bow_map.get(w,'None')
			if pol == 'Positive':
				feature_set['has_POS'+tag] = feature_set.get('has_POS'+ttags[i],0) + 1
			# Vale la pena Neutral?
			elif pol == 'Neutral':
				feature_set['has_NEU'+tag] = feature_set.get('has_NEU'+ttags[i],0) + 1
			elif pol == 'Negative':
				feature_set['has_NEG'+tag] = feature_set.get('has_NEG'+ttags[i],0) + 1
			else:
				continue
	#Presence of Negators
	negators = set(
		['no', 'nada', 'nadie', 'nunca', 'jamás', 'jamas', 'ni', 'en_mi_vida', 'ninguna', 'ninguno', 'tampoco',
		 'no_obstante', 'por_el_contrario', 'por_contra', 'antes_bien',
		 'a_pesar_de_todo', 'en_absoluto'])
	if len(negators.intersection(set(twords))) > 0:
		feature_set['has_negator'] = True

	#Presence of positive and/or negative emoticons
	positive = ['&happy', '&laugh', '&wink', '&heart', '&highfive', '&angel', '&tong']
	negative = ['&sad', '&annoyed', '&seallips', '&devil']
	if len(set(positive).intersection(set(twords))) > 0:
		feature_set['has_positiveEmoticon'] = True
	if len(set(negative).intersection(set(twords))) > 0:
		feature_set['has_negativeEmoticon'] = True
	return feature_set


#Train_SVM:
#Arguments:
#	- Classifier -> classifier model to be trained
#	- Tweets -> Set of tweets to be fed to the classifier
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


#Evaluate_Emoticons
#Arguments:
#	- tweet -> Tweet to be evaluated in base of the presence of emoticons
#Function that, given a preprocessed tweet, returns the polarity score of the same based on
#the emoticons present, if any.
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

#Find_Word:
#Receives a word and a dictionary in order to return None if the word is no in the dicitonary and
#the word itself if it iss
def find_word(word, wlist):
	result = None
	matches = [w for w in sorted(wlist) if ( w == word.encode('utf8') ) or ( w[-1]=='*' and word.startswith(w[:-1]) ) ]
	longest_match = 0
	for match in matches:
		if len(match) >  longest_match:
			longest_match = len(match)
			result = match
	return result

#Evaluate_BOW_and_features:
#Arguments:
#	- tweet -> Preprocessed tweet to be evaluated
#Function that, given a preprocessed tweet, returns the sentiment orientation of the same based
#on the sentiment of the words used and the presence of boosters, negators and intensifier.
#Constants: All taken from the works of Muhkerjee and Bhattacharyya, and Kumar and Khurana
#	-	lookup_window -> Window to be taken into account if there is presence of a negator or a intensifier
#	-	max_polarity -> Max polarity present on the dictionary. Used to invert the polarity of a word
#	-	booster_constant -> Booster constant to be applied on the booster formula
#	-	derivative -> Constant to multiply to the sentiment score of a word if it is a derivative of a root word on the dictionary
def evaluate_bow_and_features(tweet):
	lookup_window = 5 #CONSTANT
	max_polarity = 5 #CONSTANT
	booster_constant = 2 #CONSTANT
	derivative = 3	#CONSTANT
	pos_so = 0.0
	neg_so = 0.0
	intensifier = 0
	negation = False
	i_neg, i_int = 0,0
	boost_up = False
	boost_down = False
	boost_inverted = False
	#List of Words that either boost up or down the remaining of the tweet if are found
	#All these words were taken from the work of Muhkerjee and Bhattacharyya and translated to spanish and classified
	#taking into account real life uses of the words. Link to paper: http://aclweb.org/anthology//C/C12/C12-1113.pdf
	conj_inverted = ['sino']	#Inverted polarity boosters
	conj_fol = ['aún_asi', 'aun_asi', 'hasta_que', 'no_obstante']	#Boost Up Words that indicate a reinforcement of the previous sentimient
	conj_prev = ['a_pesar_de', 'aunque', 'excepto', 'salvo', 'pero', 'sin_embargo']	#Boost Down Words that indicate a contrast
	#Boost Up Words that indicate a follow up and a reinforcement of the previous orientation of the phrase
	conj_infer = ['por_lo_tanto', 'además', 'ademas', 'en_consecuencia', 'por_tanto', 'como_resultado', 'posteriormente', 'por_eso', 'hasta']
	for i, (w,tag) in enumerate(tweet):
		#Checking if the word token is in one of the set of words that boots up, down or inverted
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
			#In case the words is not on the previous sets,check if it is on the main
			#polarity dictionary
			so = 0.0
			word = find_word(w, bow_map.keys())
			#If the word is a derivative of a main word in the dictionary (Eg: Hablariamos comes for the word hablar)
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
						#Inverse polarity if there is presence of negation on the tweet
						#and if the word is on the lookup window
						if h_negation and (j-h_j_neg) <= lookup_window:
							if so > 0:
								so = so - max_polarity
							else:
								so = so + max_polarity

						#Apply intensifier if there is presence of one on the tweet
						#and if the word is on the lookup window
						if h_intensifier and (j-h_j_int) <= lookup_window:
							so = so + ( so * intensifier )

						#Add the sentiment score to the specific counter
						if so >0:
							pos_so += derivative*so
						else:
							neg_so += derivative*so

						#Check if the word is an intensifier or a negation
						if w in negator_map:
							negation = True
							h_j_neg = j
						if w in booster_map:
							intensifier = booster_map[w]
							h_j_int = j

			#If the word is not a derivative, check if is on the main dictionary
			#the word itself
			else:
				if w in bow_map:
					so = bow_map[w]
					if negation and (i-i_neg) <= lookup_window:
						if so > 0:
							so = so - max_polarity
						else:
							so = so + max_polarity
					if intensifier and (i-i_int) <= lookup_window:
						so = so + ( so * intensifier )

			#If there is a presence of a booster before the word,
			#apply the respective booster formula, all of them were taken from Kumar and Khurama
			#previous work
			if boost_up:
				so = so*booster_constant
			if boost_down:
				so = (so*(1.0))/booster_constant
			if boost_inverted:
				so = (so*1.0)/(-booster_constant)

			#Add the sentiment score to the specific counter
			if so > 0:
				pos_so += so
			else:
				neg_so += so

			#Check if the word is an intensifier or a negation
			if w in negator_map:
				negation = True
				i_neg = i
			if w in booster_map:
				intensifier = booster_map[w]
				i_int = i
	return pos_so+neg_so


#3-Layer Hybrid Classifier - Horizontal Architecture:
#   1st Layer: Emoticon Presence and Sentiment
#   2st Layer: Sentiment Orientation Calculator. Only if 1st layer evaluates to 0
#   3rd Layer: SVM. Only if all above layers return 0
#Arguments:
#  Tweet -> String
#Returns: Classification of the tweet in either Positive, Negative or Neutral category
def hybrid_classify(tweet):
	Positive = False
	Neutral = False
	Negative = False
	#First, preprocess the tweet
	filter_tweet = preprocess(tweet)

	#Evaluate the sentiment based on the emoticons present or not on the tweet
	emoticon_result = evaluate_emoticons(filter_tweet)
	if emoticon_result > 0:
		Positive = True
	elif emoticon_result < 0:
		Negative += True
	else:
		Neutral += True

	#Evaluate the sentiment based of the sentiment orientation of the words
	#For efficiency, if the sentiments matches the one thrown by the emoticon layer,
	#and it's different than neutral,return the given polarity
	bow_result = evaluate_bow_and_features(filter_tweet)
	if bow_result >= 2:
		if Positive:
			return 'Positive'
		Positive = True
	elif bow_result <= -2:
		if Negative:
			return 'Negative'
		Negative = True
	else:
		Neutral = True

	#Evaluate the sentiment based on the SVM classifier
	svm_result = svm_classifier.classify(get_features(filter_tweet))
	svm_result = svm_result.strip('\n')
	#If the result matches the orientation given by one of the
	#two layers before this, returns the matching polarity
	#In all the other cases, return the polarity given by the 2nd layer
	if svm_result == 'Positive':
		if Positive:
			return 'Positive'
		else:
			if bow_result <= -2:
				return 'Negative'
			else:
				return 'Neutral'
	elif svm_result == 'Negative':
		if Negative:
			return 'Negative'
		else:
			if bow_result >= 2:
				return 'Positive'
			else:
				return 'Neutral'
	else:
		return 'Neutral'


#Main Function
def main():
	global modifiers, booster_map, negator_map, bag_of_words
	#Creating the socket so that it can receive a tweet and send the resulting polarity
	host = ''
	port = 9999
	backlog = 5
	size = 1024
	server = socket.socket(socket.AF_INET, type=socket.SOCK_STREAM, proto=0)
	server.bind((host,port))
	server.listen(5)
	input = [server,sys.stdin]

	#Loading the dictionaries and training data
	get_pol_map()
	booster_map = get_booster_map()
	negator_map = get_negator_map()
	modifiers = get_mod_map(booster_map, negator_map)
	bag_of_words = []
	train_data = load_data_from_file(sys.argv[1])

	#Initializing and training de classifier
	global svm_classifier
	svm_classifier = SVM(type=CLASSIFICATION, kernel=LINEAR)
	train_svm(svm_classifier, train_data)

	print "Training Completed......"

	#Cycle that mantains the classifier running
	running = 1
	while running:
		inputready,outputready,exceptready = select.select(input,[],[])
		#An input is detected
		for s in inputready:

			if s == server:
				# Handle the message if it comes from a client
				#sending a connection request
				client, address = server.accept()
				input.append(client)

			elif s == sys.stdin:
				# Handle standard input, so that when entered text on the
				#console the server stops
				junk = sys.stdin.readline()
				running = 0

			else:
				# Handle all other socket connections, in this case clients sending tweets
				data = s.recv(size)
				if data:
					#Classify the tweet and send the resulting polarity
					s.send(hybrid_classify(str(data).strip("\n"))+'\r\n')
				else:
					#The client wishes to disconnect himself, close the input and remove him from the list
					s.close()
					input.remove(s)
	server.close()


if __name__ == "__main__":
	main()
