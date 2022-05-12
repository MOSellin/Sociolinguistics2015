# -*- coding: utf-8 -*-
from __future__ import division
import gensim, logging
import numpy as np
import sys
import unicodedata
import pickle
from sklearn.metrics.pairwise import pairwise_distances
import nltk
from nltk.cluster import euclidean_distance
from matplotlib.mlab import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


    def build_pca_dict(representation_dict,size):
        rep_array=np.zeros((1,size))
        for h,b in enumerate(rep_dict):
            word_rep_array=np.asarray(rep_dict[b][:size])
            rep_array=np.vstack((rep_array,word_rep_array))
        rep_array=rep_array[1:]
        a = np.array(rep_array)
        pca = PCA(a)
        a2 = np.dot(a, pca.Wt[:2].T)
        pca_dict={}
        p_count=0
        for h,b in enumerate(rep_dict):
            pca_dict[b]=a2[p_count].tolist()
            p_count+=1	
        return pca_dict	


    def vizdict(pca_dict,words):
        darepz=np.zeros((1,2))
        ext=np.zeros((1,2))
        colz=range(len(words))
        selection_labels_crazy=[]
        for s in suffixes:
            for w in words:  
                darepz=np.vstack((darepz,np.asarray(pca_dict[w+s])))
                selection_labels_crazy.append(w+s[:2])
        #label_extra=[]
        #for e in extras:
        #ext=np.vstack((ext,np.asarray(pca_dict[e])))
        for s in suffixes:
             if s in e:
                tjo=e[:-len(s)]+s[:2]
                label_extra.append(tjo)
        ext_vect=ext[1:]		
        selection_vector=darepz[1:]
        colz2=list(colz)+list(colz)
        techno=len(words)
        #plt.set_title("")
        plt.scatter(selection_vector[:,0], selection_vector[:,1], c='white')
        #plt.suptitle("Skip-gram Bamman German (PCA)",fontsize=14)
        #plt.title("Learning rate=0.1, word window=2, regularization=0.005",fontsize=10)
        #for label, x, y in zip(label_extra, ext_vect[:,0], ext_vect[:,1]):
        #	plt.text(x, y, label,color='gray',fontdict={'size': 8})
        for col, label, x, y in zip(colz2, selection_labels_crazy, selection_vector[:,0], selection_vector[:,1]):
            plt.text(x, y, label,color=plt.cm.Set1(col/techno),fontdict={'weight': 'bold', 'size': 13})
        #plt.xlim([-7.05,-5.6])
        #plt.ylim([0.7,-0.9])
        plt.show()


suffixes=['_female','_male']

#Example of visualisation of word vectors
maneder = ['april','december','juli','juni','januar','februar','marts','august','maj','november','oktober','september']	





model=sys.argv[1]
with open(model,'rb') as f:
	rep_dict=pickle.load(f)

size=int(sys.argv[2])
pca_dict0=build_pca_dict(rep_dict,size)
vizdict(pca_dict0,maneder,dage)		