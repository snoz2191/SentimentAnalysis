# -*- coding: utf-8 -*-
__author__ = 'domingo'
import sys
import locale
reload(sys)
sys.setdefaultencoding("utf-8")
import unicodedata
import re
from collections import OrderedDict
locale.setlocale(locale.LC_ALL, "es_VE.UTF-8")

f = open(sys.argv[1])
regex = re.compile(r'[\t]+')
some_dict = {}
for row in f:
	message = regex.split(row)
	word = message[0]
	word = word.strip(' \t\n\r')
	pol = message[1]
	some_dict[word] = pol
f.close()
f = open(sys.argv[2])
for row in f:
	message = regex.split(row)
	word = message[0]
	word = word.strip(' \t\n\r')
	pol = message[1]
	some_dict[word] = pol
f.close()
f = open(sys.argv[3])
for row in f:
	message = regex.split(row)
	word = message[0]
	word = word.strip(' \t\n\r')
	pol = message[1]
	some_dict[word] = pol
f.close()
f = open(sys.argv[4])
some_dict = OrderedDict(sorted(some_dict.items(), key = lambda e:locale.strxfrm(e[0])))
for elem in some_dict.items():
	s = elem[0].encode("utf8")+'\t'+elem[1].strip('\n')
	print(s)
