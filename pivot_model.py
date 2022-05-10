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


#Convert word2vec_file to dictionary	
def convert_vecfile_to_dict(model):
	dict={}
	all_reps=[]
	#Find word embeddings for each word pair
	model.wv.save_word2vec_format('pivot.model.txt', binary=False)
	veclines=codecs.open('pivot.model.txt', encoding="utf-8")
	wordlist=[]
	for i in veclines:
		line=i.split()
		if len(line)<3:
			pass
		else:	
			word=line[0]
			wordlist.append(word)
	for i in wordlist:
		dict[i]=model[i].tolist()
		all_reps.append(model[i].tolist())	
	rep_array=np.asarray(all_reps)
	average=np.average(rep_array)		
	dict['unknown']=average.tolist()
	dict['unknown_female']=average.tolist()
	dict['unknown_male']=average.tolist()	
	return dict
	
		
#Pre-processing
def preprocessing(corpora):
	num='1','2','3','4','5','6','7','8','9','0'
	regex='!','@','#','$','?','-','/',')','(','.',';',':','+','*','"',",","'",","
	processed=[]
	for review in corpora:
		review=review.lower()
		sentences = re.split(r' *[\.\?!][\'"\)\]]* *', review)
		for sentence in sentences:
			tok_sent=sentence.split()
			processedsentence=[]
			for word in tok_sent:
				word_1=word
				for r in regex:
					word_1=word_1.replace(r, "")
				for n in num:
					word_1=word_1.replace(n, "")
				if word_1=="":
					pass
				else:
					processedsentence.append(word_1)
			if len(processedsentence)==0:
				pass
			else:
				processed.append(processedsentence)				
	return processed

##Words count dictionary	
def word_count_dict(processed):
	totala={}	
	for sentence in processed:
		for word in sentence:
			if word in totala:
				totala[word] = totala[word] + 1
			else:
				totala[word] = 1			
	return totala	

##List of the count for all words frequent words	
def word_list(count_dict,topwords,cut_off):
	word_list=[]
	sorted_dict=sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)
	for tuple in sorted_dict:
		if tuple[1]>cut_off:
			word_list.append(tuple[0])
		else:
			pass
	if topwords != 0:
		word_list=word_list[:topwords]
	word_list.append('unknown')	
	return word_list
	
##Mark all words except the "num_pivots"-most frequent with suffixes for gender	
def marker_program(processed_corpora,word_list,suffix,num_pivots):
	words_to_mark=word_list[num_pivots:]
	new_list=processed_corpora
	for i,sentence in enumerate(processed_corpora):
		for j,word in enumerate(sentence):
			if word in words_to_mark:
			   new_list[i][j]=word+suffix
	return new_list	
		

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







