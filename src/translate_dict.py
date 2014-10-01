from __future__ import print_function
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

def remove_accents(input_str):
    nkfd_form = unicodedata.normalize('NFKD', unicode(input_str))
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

with open(sys.argv[1]) as f:
    regex = re.compile(r'[\s]+')
    some_dict = {}
    for row in f:
        message = regex.split(row)
        word = message[0]
        word = word.strip(' \t\n\r')
        pol = message[1]
        pol = pol.strip(' \t\n\r')
        some_dict[word] = pol
        if not is_ascii(message[0]):
                word = remove_accents(message[0])
                some_dict[word] = pol
        some_dict = OrderedDict(sorted(some_dict.items(), key = lambda e:locale.strxfrm(e[0])))
    for elem in some_dict.items():
            s = elem[0].encode("utf8")+'\t'+elem[1]
            print(s)
