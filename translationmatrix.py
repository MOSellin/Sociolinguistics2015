# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import unicode_literals
import sys
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import genfromtxt
import codecs
import gensim
import math
#import numdifftools as nd
from scipy import optimize
from matplotlib.mlab import PCA
import theano
from theano import tensor as T
from sklearn.metrics.pairwise import pairwise_distances
from theano.gof.compilelock import set_lock_status
#from tsne import *
from sklearn import (manifold, datasets, decomposition, ensemble, lda, random_projection)


set_lock_status(False)

#Check up all words from both and word pairs, which do exist in both corpora
def checkup_wordpairs(transfrom_wordlist,transto_wordlist):
	word_pairs=[]
	for word in transfrom_wordlist:
		if word in transto_wordlist:
			word_pairs.append(word)
	return word_pairs			

#Convert a vec-file into an array and get a list with all words
#in the same order as the array
def converter(vecfile):
	veclines=codecs.open(vecfile, encoding="utf-8")
	wordlist=[]
	distances=[]
	for i in veclines:
		line=i.split()
		if len(line)<3:
			pass
		else:	
			word=line[0]
			wordlist.append(word)
	return wordlist


#Find word embeddings for each word pair
def find_word_embeddings_real(translate_from_location,word_list,word_pairs,wordveclength):
	model = gensim.models.KeyedVectors.load_word2vec_format(translate_from_location, binary=False)
	embedding_vector=[]
	distance_array=np.zeros((1,wordveclength))
	for i in word_pairs: 
		distance_array=np.vstack((distance_array,model[i]))
		embedding_vector.append(i)
	distance_array=np.float32(distance_array[1:].T)
	return distance_array
	
	
#My implementation of Translation Matrix using stochastic gradient descent. Current learning rate and
#stopping criteria works fine, but could easily be adjusted
def sgd_matrix(transfrom,transto,learning_rate):
	#SET THEANO VARIABLES
	index = T.lscalar()
	x_i=T.fvector('x_i')
	z_i=T.fvector('z_i')
	x=T.fmatrix('x')
	z=T.fmatrix('z')
	print(transfrom.shape[1])
	params=theano.shared(np.random.rand(transfrom.shape[1]), name='w')
	cost_total=T.sum(abs((x*params-z)**2))
	loss=T.sum((x_i*params[index]-z_i)**2)
	d_loss_wrt_params=T.grad(loss,params)
	updates = [(params, params - learning_rate * d_loss_wrt_params)]
	SGD=theano.function([x_i,z_i,index], loss, updates=updates)
	Cost=theano.function([x,z], cost_total)
	output=theano.function([], params)
	epoch=[]
	iter=list(range(transfrom.shape[1]))
	print(type(iter))
	random.shuffle(iter)
	output=theano.function([], params)
	for i in iter:
		costiter=Cost(transfrom,transto)
		#print('Current loss is ', SGD(transfrom[:,i],transto[:,i],i))
		print('Current value of cost', costiter)
		SGD(transfrom[:,i],transto[:,i],i)
		epoch.append(costiter)
	epoch.append(1500)	
	transmat=output()		
	return transmat,params,epoch

#Outputs a translated version of the "Translated From" matrix, 
#using the translation matrix
def calculate_translation(transfrom,transmatrix):
	x=T.fmatrix('x')
	trans_calc=transmatrix*x
	translation=theano.function([x], trans_calc)
	output=translation(transfrom)
	return output
	
			
def convert_translation_transto_to_dict(wdlst,translated,transtoarray,trans_from_suffix,trans_to_suffix):
	dadict={}
	for i,word in enumerate(wdlst):
		tt_word=word+trans_to_suffix
		dadict[tt_word]=transtoarray[i].tolist() 
		tf_word=word+trans_from_suffix
		dadict[tf_word]=translated[i].tolist()
	average_tf=np.average(translated, axis=0)
	ompa='unknown'+trans_from_suffix
	dadict[ompa]=average_tf.tolist()
	average_tt=np.average(transtoarray, axis=0)
	bompa='unknown'+trans_to_suffix
	dadict[bompa]=average_tt.tolist()
	return dadict		
	
	
#IMPORT WORD2VECS	
#Specify location of Word2Vec-models
female=sys.argv[2]
male=sys.argv[1]
#Get an array with all the values from the word2vec
F_wordlist=converter(female)
M_wordlist=converter(male)
#Check word pairs that appear in both corpora
word_pairs=checkup_wordpairs(M_wordlist,F_wordlist)
word_pairs=word_pairs

word_pairs = word_pairs
#f=open('dirk_stash/danish.male.tfidf').readlines()
#word_pairs=tfidf_sorter(f,100,word_pairs)
#Find word embeddings and build arrays with all values, along with a list of all the words
M_embedding_array=find_word_embeddings_real(male,M_wordlist,word_pairs,100)
F_embedding_array=find_word_embeddings_real(female,F_wordlist,word_pairs,100)
lr=float(sys.argv[3])
		
translation_matrixnp_,translation_matrix,epoch=sgd_matrix(M_embedding_array,F_embedding_array,lr)
print (np.sum((M_embedding_array*translation_matrixnp_-F_embedding_array)**2))
translation=calculate_translation(M_embedding_array,translation_matrix)
M_embedding_arrayT=M_embedding_array.T
F_embedding_arrayT=F_embedding_array.T
translationT=translation.T
representation_dict = convert_translation_transto_to_dict(word_pairs,translation,F_embedding_arrayT,"_male","_female")
pickle.dump(representation_dict, open(sys.argv[4],"wb"))


	




##standard_female_model_danish










