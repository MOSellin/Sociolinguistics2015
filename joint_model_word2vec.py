from __future__ import division
import gensim, logging
import math
import sys
import unicodedata
from nltk import FreqDist
import operator
import cPickle as pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
from pre_processing_functions import *

#from sklearn.manifold import TSNE

    def tfidf_list_of_words(corpus):
        vectorizer = TfidfVectorizer(min_df=1)
        X = vectorizer.fit_transform(corpus)
        idf = vectorizer._tfidf.idf_
        tfids_dict=dict(zip(vectorizer.get_feature_names(), idf))
        totala_sorted=sorted(tfids_dict.items(), key=operator.itemgetter(1), reverse=True)
        return totala_sorted

    def checkup_wordpairs(wordlist_1,wordlist_2):
        word_pairs=[]
        for i in transfrom_wordlist:
            if i in transto_wordlist:
                word_pairs.append(word)
        return word_pairs	

    def compare_lists_of_measure(list,word_pairs,num_of_groups):
        measures=[]
        for i in wordpairs:
            for j in list:
                if i==j[0]:
                    measures.append(j[1])
        return measures			


    #BUILD SIMILARITY LIST BASED ON SOMETHING
    def similarity_of_word_pairs(freq_words,similarity):	
        similarity=[]	
        for i in freq_words:
            sim=model.similarity((i+end_stuff[0]),(i+end_stuff[1]))
            similarity.append(sim)
        return similarity

    def preprocessing(corpora):
        totala={}
        num='1','2','3','4','5','6','7','8','9','0'
        regex='!','@','#','$','?','-','/',')','(','.',';',':','+','*','"',",","'",","
        processed=[]
        for sentence in corpora:	
            sentence=unicode(sentence.lower(), "utf-8")
            sentence=unicodedata.normalize('NFKD', sentence).encode('ascii','ignore')
            sentence=sentence.split()
            processedsentence=[]
            for word in sentence:
                word_1=word
                for r in regex:
                    word_1=word_1.replace(r, "")
                for n in num:
                    word_1=word_1.replace(n, "")
                if word_1=="":
                    pass
                else:
                    processedsentence.append(word_1)
            for word in processedsentence:
                if word in totala:
                    totala[word] = totala[word] + 1
                else:
                    totala[word] = 1
        return totala

    def markup(corpora,freq_words,suffix):
        num='1','2','3','4','5','6','7','8','9','0'
        regex='!','@','#','$','?','-','/',')','(','.',';',':','+','*','"',",","'"
        processed=[]
        for sentence in corpora:	
            sentence=unicode(sentence.lower(), "utf-8")
            sentence=unicodedata.normalize('NFKD', sentence).encode('ascii','ignore')
            sentence=sentence.split()
            processedsentence=[]
            for word in sentence:
                word_1=word
                for r in regex:
                    word_1=word_1.replace(r, "")
                for n in num:
                    word_1=word_1.replace(n, "")
                if word_1=="":
                    pass
                else:
                    processedsentence.append(word_1)
            markedsentence=[]
            for word in	processedsentence:
                if word in freq_words:
                   markedsentence.append(word)
                else:
                    word1=word+suffix
                    markedsentence.append(word1)
            with open('danish.processed1.corpus','a') as f:
                for word in markedsentence:
                    f.write(word+" ")
                f.write('\n')


    def build_representation_matrix(end_stuff,model,word_selection):
        all_distance_array=np.zeros((len(word_selection),100,len(end_stuff)))
        for tut,end in enumerate(end_stuff):
            distance_array=np.zeros((1,100))
            for i in word_selection:
                embeddings=model[i+end]
                distance_array=np.vstack((distance_array,embeddings))		
            distance_array=np.float32(distance_array[1:])
            all_distance_array[:,:,tut]=distance_array
        return all_distance_array	




    #Freq words motsvarar cut_offs pa hogfrekventa ord samt lagfrekventa borttagna.
    #Modellen innehaller de markerade	

    """
    #CHECK LEN FOR EACH GROUP
    all=open('danish.corpus').readlines()
    print "tittut"
    tutas=preprocessing(all)
    print "tittut"
    totala_sorted = sorted(tutas.items(), key=operator.itemgetter(1), reverse=True)
    print "tittut"
    f=open('danish.Kvinde.corpus').readlines()
    markup(f,freq_words,'-f')	
    m=open('danish.Mand.corpus').readlines()
    markup(m,freq_words,'-m')	
    """
    """
    #TF-IDF CORPORA
    """
    f=open(sys.argv[1]).readlines()
    processed=[]
    for sentence in f:
        processed.append(sentence.split())	
    model=gensim.models.Word2Vec(processed, min_count=50)
    model.save_word2vec_format('text8.model.bin', binary=True)
    model.save('fmdanish_marked.model')
    model.save_word2vec_format('fmdanish_marked.model.bin', binary=True)
    model.save_word2vec_format('fmdanish_marked.model.txt', binary=False)

    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #model=gensim.models.Word2Vec.load_word2vec_format('text8.model.bin', binary=True) 










    """
    #RUN WORD2VEC
    #train word2vec on the two sentences
    #USE RELEVANT WORDS FOR ANALYSIS AND FIND DISTANCES
    #model = Word2Vec(sentences, size=200)  
    #IMPORT THE VECTORSS FOR MARKED WORDS

    with open('freq_words99999','rb') as f:
        freq_words=pickle.load(f)
    model = gensim.models.Word2Vec.load_word2vec_format('text8.model.bin', binary=True)
    print len(freq_words)[1000:]
    #RUN WORD2VEC
    #split on M/F
    #USE RELEVANT WORDS FOR ANALYSIS AND FIND DISTANCES
    #MATPLOTLIB WITH LABELS
    #Plot the classifications	




    #GET ALL JOINT WORDS AND VISUALIZE A BUNCH OF THEM
    end_stuff='-m','-f'

    def build_representation_matrix(end_stuff,model,word_selection):
        all_distance_array=np.zeros((len(word_selection),100,len(end_stuff)))
        for tut,end in enumerate(end_stuff):
            distance_array=np.zeros((1,100))
            for i in word_selection:
                embeddings=model[i+end]
                distance_array=np.vstack((distance_array,embeddings))		
            distance_array=np.float32(distance_array[1:])
            all_distance_array[:,:,tut]=distance_array
        return all_distance_array
    """













#BUILD TSNE
"""
#model = TSNE(n_components=2, random_state=0)
#model.fit_transform(X) 
"""
    #DO PARALELL WORD PROCESSING	



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







