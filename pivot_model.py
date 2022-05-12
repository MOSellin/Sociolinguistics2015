#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import itertools
from collections import OrderedDict
import numpy as np
import unicodedata
from heapq import heappush, heappop, heapify
from collections import defaultdict
import operator
import pickle
import sys
from operator import itemgetter
import re
import codecs
import pickle
import gensim
from pre_processing_functions import *
 
## Import data
m=codecs.open(sys.argv[1], encoding="utf-8")
f=codecs.open(sys.argv[2], encoding="utf-8")
f_proc=preprocessing(f)
m_proc=preprocessing(m)

#Process and mark the whole corpora for the future word2vec model
total=m_proc+f_proc
word_dict=word_count_dict(total)
wordlist=word_list(word_dict,700,15)
total = None
f_marked=marker_program(f_proc,wordlist,'_female',75)
m_marked=marker_program(m_proc,wordlist,'_male',75)
f_proc = None
m_proc = None
all_marked=f_marked+m_marked

#build model and save it as a dictionary in a pickle file 
model=gensim.models.Word2Vec(all_marked, min_count=25)
representation_dict=convert_vecfile_to_dict(model)
pickle.dump(representation_dict, open(sys.argv[3],"wb"))







