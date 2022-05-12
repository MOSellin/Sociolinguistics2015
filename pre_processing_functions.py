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
rep_maker,shared_dataset,shared_range,parameters,theano_p_y_given_x,
get_one_prob_per_path,negative_log_likelihood,preprocessing,word_list
preprocessing,word_list




#Build dictionary based on the trained word embeddings
def rep_maker(word_weights,bamman_layer,bammandict,wordlist,neuraldict):
    bamman_representations={}
    bk=T.lscalar()
    indexx=T.lscalar()
    j_o=theano.function([indexx],word_weights[indexx])
    new=theano.function([indexx,boink],bamman_layer[indexx,bk])
    for bamvar in bammandict:
        suffix=bammandict[bamvar]
        for word in wordlist:
            neuron=neuraldict[word]
            ww=j_o(neuron)
            bw=new(neuron,bamvar)
            new_part=np.hstack((ww,bw))	
            new_part_list=new_part.tolist()
            rep_name=word+'_'+suffix
            bamman_representations[rep_name]=new_part_list
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

#Function which is part of building Huffman tree
def shared_range(window_size):
    update_range=np.arange(window_size*2)
    print(update_range.shape)
 shared_update_range=theano.shared(np.asarray(update_range,dtype=theano.config.floatX),borrow=True)
    return T.cast(shared_update_range, 'int32')

def parameters(insert_learning_rate_here): learning_rate=theano.shared(value=np.array(insert_learning_rate_here,dtype=theano.config.floatX),name='row_i',borrow=True)
    row_i=theano.shared(value=np.array(-1,dtype=theano.config.floatX),name='row_i',borrow=True)
    y_j=theano.shared(value=np.array(0,dtype=theano.config.floatX),name='y_j',borrow=True)
    return row_i,y_j,learning_rate

#Function which is part of building Huffman tree
def theano_p_y_given_x(path_,t_n,dttn):
    new_shape=T.stack(200,2)
    etttten=T.reshape(t_n,new_shape,ndim=2)
    inputinput = T.dot(dttn, etttten) + b
    yy = T.nnet.softmax(inputinput)
    tjohej = yy[0][path_]
    return tjohej

#Function which is part of building Huffman tree
def get_one_prob_per_path(node_list,path_list,length,xx,tree_nodas,uniquelist):
    nodes=T.as_tensor_variable(node_list[0:[length]])
    paths=T.as_tensor_variable(path_list[0:[length]])	
    new_nodes,_ii_=theano.scan(fn = rebuilder_guy,sequences=[nodes],non_sequences=[uniquelist])
    ww=tree_nodas[new_nodes]
    prob_prob,_upd_=theano.scan(fn = theano_p_y_given_x,sequences=[paths,ww],non_sequences=[xx])
    one_prob,__upd__=theano.reduce(lambda x, y: x*y, sequences=prob_prob, outputs_info=None, non_sequences=temp)
    return one_prob

#Function which is part of building Huffman tree
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

#Pre-processing of reviews
def build_bigrammer(data,collo):
    processed=[]
    for sentence in data:
        collo_processed=collo[sentence]
        processed.append(collo_processed)
    return processed

#Word count	
def word_count_dict(processed):
    totala={}
    for sentence in processed:
        for word in sentence:
            if word in totala:
                totala[word] = totala[word] + 1
            else:
                totala[word] = 1
    return totala

#Builds list with most frequent words
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

#Dictionary for representing words in Theano
def neural_dict(word_list):
    length=len(word_list)
    neural_dict={}
    for i in range(length):
        neural_dict[word_list[i]]=i
    neural_dict['unknown']=len(word_list)
    return neural_dict	
    
#Transform words to Huffman tree
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
    
#Sort Huffman dict
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

#Build Huffman dictionary
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
    binaries_1=['1','0']
    binaries_2=['1','0']
    for i in range(maxlength):
        add=[]
        takeoff=[]
        for j in binaries_1:
            if j in input_nodes:
                takeoff.append(j)
            else:
                tree_nodes.append(j)
                for k in binaries_2:
                    stuff=j+k
                    add.append(stuff)
                takeoff.append(j)	
        for k in takeoff:
            binaries_1.remove(k)
        for l in add:
            binaries_1.append(l)
    return input_nodes,tree_nodes


#Create paths for y variable
def get_y_paths(neuron,huff_dict,tree_nodes):
    paths=[]
    nodes=[]
    y_huffs=huff_dict[neuron]
    path_temp=list(str(y_huffs))
    path_temp=path_temp[2:-2]
    tut=''
    nodes.append(0)
    for k,j in enumerate(path_temp):
        tut=tut+j
        if k==(len(path_temp)-1):
            pass
         else:
            nodes.append(tree_nodes.index(tut))
        paths.append(int(float(j)))
    return paths,nodes

#Build node path array
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
    shared_paths_array = theano.shared(np.asarray(paths_array,dtype=theano.config.floatX),borrow=True)
    shared_nodes_array = theano.shared(np.asarray(nodes_array,dtype=theano.config.floatX),borrow=True)
    shared_lengths_array = theano.shared(np.asarray(lengths_array,dtype=theano.config.floatX),borrow=True)	
    return T.cast(shared_paths_array, 'int32'),T.cast(shared_nodes_array, 'int32'),T.cast(shared_lengths_array, 'int32')


#Build neural model from neural dictionary    
def neural_model(neural_dict,sentences,window_size,gt):
    temp_range=range(0,len(sentences),100)
    temp_arr_two=np.zeros((1,(2+2*window_size)))
    temp_arr=np.zeros((10000,(2+2*window_size)))
    temp_cout=0
    for z,x in enumerate(temp_range):
        if x==0:
            part_sentences=sentences[temp_range[-1]:]
        else:
            part_sentences=sentences[temp_range[z-1]:x]
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
    output_=temp_arr_two[1:]
    output_arr=np.array(output_arr, dtype=np.int32)
    return output_arr

set_lock_status(False)

@as_op(itypes=[T.imatrix],
       otypes=[T.ivector])
#Transform to integet
def numpy_unique(a):
    to_int=np.unique(a)
    to_int_1=to_int.astype(np.int32, copy=False)
    return to_int_1


#Rebuild a node matrix
@as_op(itypes=[T.iscalar,T.ivector],
       otypes=[T.ivector])
def rebuilder_guy(one_node,local_node_list):
    t_o=np.where(local_node_list == one_node)
    outs=t_o[0].astype(np.int32, copy=False)
    return outs
    
#Retrieve all relevant nodes
def get_all_relevant_nodes(i_train):
    y_i=y[i_train]
    allnodesmatrix=shared_nodes_array[y_i]
    tjoas=numpy_unique(a)
    return tjoas


#Translation Matrix
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
    
    
#My implementation of Translation Matrix using stochastic gradient descent. 
def sgd_matrix(transfrom,transto,learning_rate):
    #SET THEANO VARIABLES
    index = T.lscalar()
    x_i=T.fvector('x_i')
    z_i=T.fvector('z_i')
    x=T.fmatrix('x')
    z=T.fmatrix('z')
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

#Translation to dictionary
def convert_translation_transto_to_dict(wdlst,translated,transtoarray,trans_from_suffix,trans_to_suffix):
    d_dict={}
    for i,word in enumerate(wdlst):
        tt_word=word+trans_to_suffix
        d_dict[tt_word]=transtoarray[i].tolist() 
        tf_word=word+trans_from_suffix
        d_dict[tf_word]=translated[i].tolist()
    average_tf=np.average(translated, axis=0)
    first='unknown'+trans_from_suffix
    d_dict[first]=average_tf.tolist()
    average_tt=np.average(transtoarray, axis=0)
    second='unknown'+trans_to_suffix
    d_dict[second]=average_tt.tolist()
    return dadict


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
def preprocessing_m(corpora):
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
