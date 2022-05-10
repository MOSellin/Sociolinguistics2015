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
	
#Build dictionary based on the trained word embeddings
def rep_maker(word_weights,bamman_layer,bammandict,wordlist,neuraldict):
	bamman_representations={}
	bk=T.lscalar()
	indexx=T.lscalar()
	jorma=theano.function([indexx],word_weights[indexx])
	nyttan=theano.function([indexx,boink],bamman_layer[indexx,bk])
	for bamvar in bammandict:
		suffix=bammandict[bamvar]
		for word in wordlist:
			neuron=neuraldict[word]
			ww=jorma(neuron)
			bw=nyttan(neuron,bamvar)
			nygrej=np.hstack((ww,bw))	
			nygrej_list=nygrej.tolist()
			rep_name=word+'_'+suffix
			bamman_representations[rep_name]=nygrej_list
	return bamman_representations


	
##Puts all relevant data and parameters into the Theano computation graph	
def shared_dataset(datan, borrow=True):
    geo=datan[:,0]
    x=datan[:,1]
    y=datan[:,2:]
    shared_geo = theano.shared(np.asarray(geo,dtype=theano.config.floatX),borrow=borrow)
    shared_x = theano.shared(np.asarray(x,dtype=theano.config.floatX),borrow=borrow)
    shared_y = theano.shared(np.asarray(y,dtype=theano.config.floatX),borrow=borrow)
    return T.cast(shared_geo, 'int32'),T.cast(shared_x, 'int32'),T.cast(shared_y, 'int32')

def shared_range(window_size):
    update_range=np.arange(window_size*2)
    print(update_range.shape)
    shared_update_range=theano.shared(np.asarray(update_range,dtype=theano.config.floatX),borrow=True)
    return T.cast(shared_update_range, 'int32')

def parameters(insert_learning_rate_here):
	learning_rate=theano.shared(value=np.array(insert_learning_rate_here,dtype=theano.config.floatX),name='row_i',borrow=True)
	row_i=theano.shared(value=np.array(-1,dtype=theano.config.floatX),name='row_i',borrow=True)
	y_j=theano.shared(value=np.array(0,dtype=theano.config.floatX),name='y_j',borrow=True)
	return row_i,y_j,learning_rate


def theano_p_y_given_x(path_,tuten,datten):
	new_shape=T.stack(200,2)
	ditten=T.reshape(tuten,new_shape,ndim=2)
	inputinput = T.dot(datten, ditten) + b
	yy = T.nnet.softmax(inputinput)
	tjohej = yy[0][path_]
	return tjohej
		
def get_one_prob_per_path(node_list,path_list,length,xx,tree_nodas,uniquelist):
	nodes=T.as_tensor_variable(node_list[0:[length]])
	paths=T.as_tensor_variable(path_list[0:[length]])	
	new_nodes,_ii_=theano.scan(fn = rebuilder_guy,sequences=[nodes],non_sequences=[uniquelist])
	ww=tree_nodas[new_nodes]
	probi_probi,_upd_=theano.scan(fn = theano_p_y_given_x,sequences=[paths,ww],non_sequences=[xx])
	one_prob,__upd__=theano.reduce(lambda x, y: x*y, sequences=probi_probi, outputs_info=None, non_sequences=temp)
	return one_prob

		
def negative_log_likelihood(y_i,xx):
	return -T.mean(T.log(allpaths(y_i,xx))[T.arange(y_i.shape[0]), y])	
	
	
#Pre-processing of reviews
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

def build_bigrammer(data,collo):
	processed=[]
	for sentence in data:
		collo_processed=collo[sentence]
		processed.append(collo_processed)
	return processed

#WORD COUNT	
def word_count_dict(processed):
	totala={}
	for sentence in processed:
		for word in sentence:
			if word in totala:
				totala[word] = totala[word] + 1
			else:
				totala[word] = 1			
	return totala	

#CUT OF WORDS WE DONT WANT TO USE AND BUILD A LIST OF WORDS
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

def neural_dict(word_list):
	length=len(word_list)
	neural_dict={}
	for i in range(length):
		neural_dict[word_list[i]]=i
	#neural_dict['unknown']=len(word_list)	
	return neural_dict	
	
def words_to_neural_huffman(word_list,neural_dict,count_dict):
	huffman_dict={}
	huffman_dict[(len(word_list)-1)]=0
	for i in count_dict:
		if i in word_list:	
			neural_number=neural_dict[i]
			count=count_dict[i]
			huffman_dict[neural_number]=count
		else:
			huffman_dict[(len(word_list)-1)] = huffman_dict[(len(word_list)-1)] + count_dict[i]				
	return huffman_dict
	
def encode(symb2freq):
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def build_huffman_dict(huffman_list):
	huff_dict={}
	for i in huffman_list: 
		huff_dict[i[0]]=i[1:]
	return huff_dict
	
#Sort the list of binaries
def build_node_weight_array(huffman_list):
	maxlength=len(huffman_list[-1][1])
	input_nodes=[]
	for i in huffman_list:
		node=i[1]
		input_nodes.append(node)
	tree_nodes=[]
	tree_nodes.append('start')
	bina=['1','0']
	bino=['1','0']
	for i in range(maxlength):
		add=[]
		takeoff=[]
		for j in bina:
			if j in input_nodes:
				takeoff.append(j)
			else:
				tree_nodes.append(j)
				for k in bino:
					stuff=j+k
					add.append(stuff)
				takeoff.append(j)	
		for k in takeoff:
			bina.remove(k)
		for l in add:
			bina.append(l)
	return input_nodes,tree_nodes 	
	
	
def get_y_paths(neuron,huff_dict,tree_nodes):
	paths=[]
	nodes=[]
	y_huffs=huff_dict[neuron]
	tjoas=list(str(y_huffs))
	tjoas=tjoas[2:-2]
	tut=''
	nodes.append(0)
	for k,j in enumerate(tjoas):
		tut=tut+j
		if k==(len(tjoas)-1):
			pass
		else:	
			nodes.append(tree_nodes.index(tut))
		paths.append(int(float(j)))
	return paths,nodes
	
def build_path_node_array(wordlist,huff_dict,tree_nodes):
	maxlength=1
	paths_array=np.zeros((len(wordlist),25))
	nodes_array=np.zeros((len(wordlist),25))
	lengths_array=np.zeros((len(wordlist),1))
	for i,word in enumerate(wordlist):
		neuron=neurodict[word]
		paths,nodes=get_y_paths(neuron,huff_dict,tree_nodes)
		length=len(paths)
		lengths_array[i]=length
		paths_array[i,:length]=paths
		nodes_array[i,:length]=nodes
		if length > maxlength:
			maxlength=length
	paths_array=paths_array[:,:maxlength]
	nodes_array=nodes_array[:,:maxlength]
	print(nodes_array.shape)
	shared_paths_array = theano.shared(np.asarray(paths_array,dtype=theano.config.floatX),borrow=True)
	shared_nodes_array = theano.shared(np.asarray(nodes_array,dtype=theano.config.floatX),borrow=True)
	shared_lengths_array = theano.shared(np.asarray(lengths_array,dtype=theano.config.floatX),borrow=True)	
	return T.cast(shared_paths_array, 'int32'),T.cast(shared_nodes_array, 'int32'),T.cast(shared_lengths_array, 'int32')

	
def neural_model(neural_dict,sentences,window_size,gt):
	temp_range=range(0,len(sentences),100)
	temp_arr_two=np.zeros((1,(2+2*window_size)))
	temp_arr=np.zeros((10000,(2+2*window_size)))
	temp_cout=0
	for z,hejhopp in enumerate(temp_range):
		if hejhopp==0:
			part_sentences=sentences[temp_range[-1]:]
		else:
			part_sentences=sentences[temp_range[z-1]:hejhopp]
		for sentence in part_sentences:
			neural_string=[]
			for i in range(window_size):
				neural_string.append(neural_dict['unknown'])
			for word in sentence:
				if word in wordlist:
					number=neural_dict[word]
					neural_string.append(number)
				else:
					number=neural_dict['unknown']	
					neural_string.append(number)
		#Word window=two first, two last
			string_length=len(neural_string)
			for i in range(window_size,(string_length-window_size)):
				temp_arr[temp_cout,0]=gt
				temp_arr[temp_cout,1]=neural_string[i]
				x_before=neural_string[(i-window_size):i]
				x_after=neural_string[i:(i+window_size)]
				x_total=x_before+x_after
				temp_arr[temp_cout,2:]=x_total
				temp_cout+=1
				if temp_cout==10000:
					temp_arr_two=np.vstack((temp_arr_two,temp_arr))
					temp_arr=np.zeros((10000,(2+2*window_size)))
					temp_cout=0
	temp_arr_two=np.vstack((temp_arr_two,temp_arr[:temp_cout]))
	#EDIT THIS SHIT
	outputte_vickman=temp_arr_two[1:]
	outputte_vickman=np.array(outputte_vickman, dtype=np.int32)
	return outputte_vickman	

set_lock_status(False)
	
	
@as_op(itypes=[T.imatrix],
       otypes=[T.ivector])
def numpy_unique(a):
	tjoas=np.unique(a)
	#change it
	hejas=tjoas.astype(np.int32, copy=False)
	return hejas	
	

#Rebuild a node matrix
@as_op(itypes=[T.iscalar,T.ivector],
       otypes=[T.ivector])
def rebuilder_guy(one_node,local_node_list):
	tjo=np.where(local_node_list == one_node)
	strulas=tjo[0].astype(np.int32, copy=False)
	return strulas
	 
#Alternate
def get_all_relevant_nodes(i_train):
	y_i=y[i_train]
	allnodesmatrix=shared_nodes_array[y_i]
	tjoas=numpy_unique(a)
	return tjoas
	

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


#uild wordlists, vectors and huffman_trees for the skip-gram algortihm
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
outputtevickman_male=None 
outputtevickman_female=None 
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
#node_list,path_list,length,xx,tree_nodas,uniquelist
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
		
#Transform into dictionary, then puts it in a pickle file
bammandict={}
bammandict[0]='male'
bammandict[1]='female'
representation_dict=rep_maker(word_weights,bamman_layer,bammandict,wordlist,neurodict)
pickle.dump(representation_dict, open(sys.argv[3],"wb")) 		
