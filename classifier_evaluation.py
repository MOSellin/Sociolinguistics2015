from __future__ import division
from numpy import zeros as np_zeros
from numpy import hstack as np_hstack
from numpy import asarray as np_asarray
from numpy import vstack as np_vstack
from numpy import amin as np_amin
from numpy import amax as np_amax
from numpy import average as np_average
from collections import OrderedDict
import unicodedata
from sklearn.metrics.pairwise import pairwise_distances
from heapq import heappush, heappop, heapify
from collections import defaultdict
from theano import tensor as T
import operator
import pickle
import sys
from operator import itemgetter
import re
import glob, sys
import codecs
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from gensim import models
import gensim
from sklearn.metrics import f1_score

#Pre-processing
def ppbigram(corpora):
	regex='!','@','#','$','?','-','_','/',')','(','.',';',':','+','*','"',",","'",",",'&','='
	p_review=[]
	tut=corpora	
	processedreview=[]
	totalas=[]
	start=0
	for i,sentence in enumerate(corpora):
		if len(processedreview)>0:
			for sentence in processedreview:
				totalas.append(sentence) 
				processedreview=[]
		else:   
			processedsentence=[]
			lowsent=sentence.lower()
			tok_sent=lowsent.split()
			sent_only=tok_sent[1:]
			for word in sent_only:
				word_1=word
				for r in regex:
					word_1=word_1.replace(r, "")
				if word_1=="":
					pass
				else:
					processedsentence.append(word_1)
			if len(processedsentence)==0:
				pass
			else:
				processedreview.append(processedsentence)
	bigram.add_vocab(totalas)	

	
	
#Rebuild reviews for classification	using word embeddings
def tangs_method_suffix(inputreview,suffix,representation_dict):
	sent_rep=[]
	for sentence in inputreview:
		for word in sentence:
			word_with_suffix=word+suffix
			if word_with_suffix not in representation_dict:
				word_with_suffix='unknown'+suffix	
			#else:
			#	print word
			rep_list=representation_dict[word_with_suffix]
			sent_rep.append(rep_list)
	rep_array=np_asarray(sent_rep)
	min=np_amin(rep_array, axis=0)
	max=np_amax(rep_array, axis=0)
	average=np_average(rep_array, axis=0)
	average_rep=np_hstack((min,average,max))
	average_rep=np_hstack((min,average,max))
	return average_rep

def tangs_method_nosuffix(inputreview,representation_dict):
	sent_rep=[]
	for sentence in inputreview:
		for word in sentence:
			if word in representation_dict:
				rep_list=representation_dict[word]
			else:
				rep_list=representation_dict['unknown']
			sent_rep.append(rep_list)	
	rep_array=np_asarray(sent_rep)
	min=np_amin(rep_array, axis=0)
	max=np_amax(rep_array, axis=0)
	average=np_average(rep_array, axis=0)
	average_rep=np_hstack((min,average,max))
	return average_rep	

#Rebuild reviews from scratch
def build_dataset(corpora,suffix,rep_dict,size):
#Pre-processing
	regex='!','@','#','$','?','-','_','/',')','(','.',';',':','+','*','"',",","'",",","&"
	p_review=[]
	thelines=codecs.open(corpora, encoding="utf-8")
	processed=[]
	labels_out=[]
	labels=[]
	processedreview=[]
	reps=np_zeros((1,(size*3)))
	temp_matrix=np_zeros((10000,(size*3)))
	tutas=0
	for i,sentence in enumerate(thelines):
		if len(processedreview)>0:
			if suffix=='no':
				hej=tangs_method_nosuffix(processedreview,rep_dict)
			else:
				hej=tangs_method_suffix(processedreview,suffix,rep_dict)
			temp_matrix[tutas]=hej
			tutas+=1
			processedreview=[]
			labels_out.append(labels[0])
			labels=[]
			if tutas==10000:
				reps=np_vstack((reps,temp_matrix))
				temp_matrix=np_zeros((10000,(size*3)))
				tutas=0
		else:
			processedsentence=[]
			tok_sent=sentence.split()
			if len(sentence)>1:
				labels.append(tok_sent[0])
			if len(sentence)>1:
				sent_only=tok_sent[1:]
				for word in sent_only:
					word_1=word
					for r in regex:
						word_1=word_1.replace(r, "")
					if word_1=="":
						pass
					else:
						processedsentence.append(word_1)
			if len(processedsentence)==0:
				pass
			else:
				collo=bigram[processedsentence]
				if len(collo)>2:
					processedreview.append(collo)			
	reps=np_vstack((reps[1:],temp_matrix[:tutas]))
	return labels_out,reps	

def build_datasets(data,rep_dict,size):
	trainlabels=[]
	traindata1=np_zeros((1,(size*3)))
	testlabels=[]
	testdata1=np_zeros((1,(size*3)))
	devlabels=[]
	devdata1=np_zeros((1,(size*3)))
	for data in splitz:
		suffix=suffix_output(data,suffixas)			
		if 'train' in data:
			print(data)
			labels,reps=build_dataset(data,suffix,rep_dict,size)
			for i in labels:
				trainlabels.append(i)
			print(reps.shape)	
			traindata1=np_vstack((traindata1,reps))	
		if 'test' in data:
			labels,reps=build_dataset(data,suffix,rep_dict,size)
			for i in labels:
				testlabels.append(i)
			testdata1=np_vstack((testdata1,reps))	
		if 'dev' in data:
			labels,reps=build_dataset(data,suffix,rep_dict,size)
			for i in labels:
				devlabels.append(i)	
			devdata1=np_vstack((devdata1,reps))
	traindata1=traindata1[1:]
	testdata1=testdata1[1:]
	devdata1=devdata1[1:]	
	return traindata1,testdata1,devdata1,trainlabels,testlabels,devlabels
	
#Recognize Build suffixes	
def suffix_output(data,suffixas):
	suffix='billy'
	if '.M.' in data and suffixas=='gender':
		suffix='_male'
	if '.F.' in data and suffixas=='gender':
		suffix='_female'	
	if '.0.' in data and suffixas=='age':
		suffix='_age_01'
	if '.1.' in data and suffixas=='age':
		suffix='_age_02'
	if '.2.' in data and suffixas=='age':
		suffix='_age_03'
	if suffixas=='nosuffix':
		suffix='no'
	return suffix



	
	
print("started")
datadirectory=sys.argv[1]
splitz=glob.glob((datadirectory+'*'))	
print("started")

#define classifier
clf = LogisticRegression()


#values='gender','age','nosuffix'
suffixas=sys.argv[3]

print("start preprocessing")
#Extra preprocessing
bigram=models.Phrases(['one','trial','round','here','and','now'])	
for d in splitz:
	if 'train' in d:
		stuff=codecs.open(d, encoding="utf-8")
		ppbigram(stuff)
print("started")
		

thesize=int(sys.argv[4])
model=sys.argv[2]
print("start classification")
with open(model,'rb') as f:
	rep_dict=pickle.load(f)
print(model)
traindata,testdata,devdata,trainlabels,testlabels,devlabels=build_datasets(splitz,rep_dict,thesize)
print(traindata.shape)
print("start logreg")
clf.fit(traindata,trainlabels)
predz=clf.predict(devdata)
f1=f1_score(devlabels, predz, average='weighted')
print("F1-score weighted")
print(f1)
f1=f1_score(devlabels, predz, average='micro')
print("F1-score micro")
print(f1)
f1=f1_score(devlabels, predz, average='macro')
print("F1-score macro")
print(f1)
print("accuracy")
print(clf.score(devdata,devlabels))








