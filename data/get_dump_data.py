# -*- coding: utf-8 -*-
__author__ = 'domingo'
import sys
import locale
reload(sys)
sys.setdefaultencoding("utf-8")
locale.setlocale(locale.LC_ALL, "es_VE.UTF-8")
import json

file = open(sys.argv[2], 'w+')
text = open('./DUMP.txt').read()
list = json.loads(text)
for elem in list:
	text = elem['text']
	posivote = elem['posivote']
	neuvote = elem['neuvote']
	negvote = elem['negvote']
	if not (posivote or neuvote or negvote):
		continue
	else:
		if posivote != 0:
			file.write(text.encode('utf8') + '\tPositive')
			file.write('\n')
		elif neuvote != 0:
			file.write(text.encode('utf8') + '\tNeutral')
			file.write('\n')
		elif negvote != 0:
			file.write(text.encode('utf8') + '\tNegative')
			file.write('\n')
file.close()