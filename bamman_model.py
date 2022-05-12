# -*- coding: utf-8 -*-
from __future__ import division
import codecs
import itertools
from collections import OrderedDict
import theano;theano.__version__
import numpy as np
from theano.compile.ops import as_op
from theano.gof.compilelock import set_lock_status
from heapq import heappush, heappop, heapify
from collections import defaultdict
from theano import tensor as T
import operator
import pickle
import sys
from operator import itemgetter
import re
from gensim import models
from pre_processing_functions import *



window_size=int(sys.argv[4])
learnrate=float(sys.argv[5])
lambda_=float(sys.argv[6])

##f=codecs.open(sys.argv[1], encoding="utf-8")
##m=codecs.open(sys.argv[2], encoding="utf-8")



#Import corpora and preprocesses data
f=codecs.open(sys.argv[1], encoding="utf-8")
m=codecs.open(sys.argv[2], encoding="utf-8")
f_proc=preprocessing(f)
m_proc=preprocessing(m)
f_proc=[item for sublist in f_proc for item in sublist]	
bigram=models.Phrases(f_proc)
m_proc=[item for sublist in m_proc for item in sublist]	
bigram.add_vocab(m_proc)
f_proc=build_bigrammer(f_proc,bigram)
m_proc=build_bigrammer(m_proc,bigram)
print("processed")
total=f_proc+m_proc
print (len(total))
dikto=word_count_dict(total)
print("dikto")


#Build wordlists, vectors and huffman_trees for the skip-gram algortihm
wordlist=word_list(dikto,0,50)
total=None
neurodict=neural_dict(wordlist)


#builds Huffman codes and recodes the dataset into a neural model
huffman_input=words_to_neural_huffman(wordlist,neurodict,dikto)
neuron_to_huffman=encode(huffman_input)
list_of_input_nodes,list_of_parent_nodes=build_node_weight_array(neuron_to_huffman)
huffman_dict=build_huffman_dict(neuron_to_huffman)
output_male=neural_model(neurodict,m_proc,window_size,0)
output_female=neural_model(neurodict,f_proc,window_size,1)
print(output_male.shape)
print(output_female.shape)
train_data=np.vstack((output_male,output_female))
print(train_data.shape)
np.random.shuffle(train_data)

#Delete all none-used files before calculation

f_proc=None
m_proc=None


#Models specifications
bamman_variable=1
bamman_layers=2
tree_array_length=len(list_of_parent_nodes)	
print("wordlist")
print(len(wordlist))		
all_input_nodes=len(list_of_input_nodes)
print(all_input_nodes)

whole_length=train_data.shape[0]
print(whole_length)


#geo,x,y=shared_dataset(train_data)
b=np.zeros(2)
b[1]=1
train_data=train_data

#Specifications for Theano put for the skip-gram model
#Shared weights
update_range=shared_range(window_size)
shared_paths_array,shared_nodes_array,shared_lengths_array=build_path_node_array(wordlist,huffman_dict,list_of_parent_nodes)
word_weights=theano.shared(value=np.random.rand(all_input_nodes,100), name='word_weights', borrow=True)
bamman_layer=theano.shared(value=np.random.rand(all_input_nodes,bamman_layers,100), name='bamman_layer', borrow=True)
tree_node_weights=theano.shared(value=np.random.rand(tree_array_length,((bamman_variable+1)*100),2), name='tree node weights', borrow=True)
b=theano.shared(value=b,name='bias',borrow=True)
temp=theano.shared(value=np.ones((1),dtype=theano.config.floatX),name='temp scal',borrow=True) ##Should be floats
L2=theano.shared(value=np.array(0,dtype=theano.config.floatX),name='temp scal',borrow=True)

#Definition of input matrixes
#i_train=T.lscalar('i_train')
x=T.iscalar('target')
y=T.ivector('context')
geo=T.lscalar('gender')
learning_rate=T.dscalar('tolearn')
lambda_reg=T.dscalar('reg_code')
index=x
y_i=y
bammanindex=geo
allnodesmatrix=shared_nodes_array[y_i]
alluniquenodes=numpy_unique(allnodesmatrix)
ww=T.as_tensor_variable(tree_node_weights[alluniquenodes])
wordwi=word_weights[[index]] 
bammvi=bamman_layer[[index],[bammanindex]]


#All nodes we want to use
local_nodes=shared_nodes_array[y_i]
#All paths we want to use
local_paths=shared_paths_array[y_i]
local_lengths=shared_lengths_array[y_i]
xx=T.as_tensor_variable(T.concatenate([wordwi, bammvi], axis=1)) 
results_,updates=theano.scan(fn=get_one_prob_per_path,sequences=[local_nodes,local_paths,local_lengths],non_sequences=[xx,ww,alluniquenodes])
ll=-T.mean(T.log(results_))

#Loss function
lost=ll+L2

#Calculate gradients
xx_grad=T.grad(lost,wordwi)
bam_grad=T.grad(lost,bammvi)
ww_grad=T.grad(lost,ww)

#Updates the parameters in the computation graph
total_update=[(tree_node_weights,T.set_subtensor(tree_node_weights[alluniquenodes],tree_node_weights[alluniquenodes]-learning_rate*ww_grad)),
(word_weights,T.set_subtensor(word_weights[[index]],word_weights[[index]]-learning_rate*xx_grad)),
(bamman_layer,T.set_subtensor(bamman_layer[[index],[bammanindex]],bamman_layer[[index],[bammanindex]]-learning_rate*bam_grad))]
SGD_total=theano.function([x,y,geo,learning_rate], lost, updates=total_update)
sumparameters=T.sum(bamman_layer**2)+T.sum(word_weights**2)
L2update=[(L2,(sumparameters*lambda_reg))]
updateL2=theano.function([lambda_reg],sumparameters,updates=L2update)

print("Run SGD Skip-Gram in Theano")
check_up=range(0,60000000,100000)
iteras=range(0,60000000,10000)
for i in range(whole_length):
    if i in check_up:
        print(i)
        updateL2(lambda_)
    observation=train_data[i]
    gender=observation[0]
    target=observation[1]
    context=observation[2:]
    SGD_total(target,context,gender,learnrate)
    if i in iteras:
        print(i)

#Transform into dictionary, then put it into a pickle file
bammandict={}
bammandict[0]='male'
bammandict[1]='female'
representation_dict=rep_maker(word_weights,bamman_layer,bammandict,wordlist,neurodict)
pickle.dump(representation_dict, open(sys.argv[3],"wb")) 		
