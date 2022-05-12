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
from pre_processing_functions import *


set_lock_status(False)


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









