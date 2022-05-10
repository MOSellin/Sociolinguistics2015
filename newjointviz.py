# -*- coding: utf-8 -*-
from __future__ import division
import gensim, logging
import numpy as np
import sys
import unicodedata
import codecs
import random
from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#IMPORT VECFILE WITH WORDLIST
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
	allwords=[]
	allwords_2=[]
	for word in wordlist:
		if suffixes[0] in word:			
			allwords.append(word[:-(len(suffixes[0]))])
		if suffixes[1] in word:			
			allwords_2.append(word[:-(len(suffixes[1]))])	
	callwords=[]
	for i in allwords:
		if i in allwords_2:
			callwords.append(i)	
	return callwords	   

	

def converterX(vecfile):
	model = gensim.models.Word2Vec.load_word2vec_format(vecfile, binary=False)
	stefan=codecs.open(vecfile, encoding="utf-8")
	dadict={}
	crazy=[]
	all=np.zeros((1,100))
	words=[]
	for line in stefan:
		tjo=line.split()
		if len(tjo)<3:
			pass
		else:	
			words.append(tjo[0])
			word_embedding=model[tjo[0]]
			all=np.vstack((all,word_embedding))
	rep_array=all[1:]		
	return words,rep_array	
	

#BUILD SIMILARITY LIST BASED ON SOMETHING
def similarity_of_word_pairs(allwords,vecfile):
	model = gensim.models.Word2Vec.load_word2vec_format(vecfile, binary=False)	
	similarity=[]	
	for i in allwords:
		sim=model.similarity((i+suffixes[0]),(i+suffixes[1]))
		similarity.append(sim)
	bigosz=[]
	storan=[]
	lillan=[]
	sims=[]
	for j in range(len(allwords)):
		word=allwords[j]
		sim=similarity[j]
		sims.append(sim)
		if sim<0.50:
		   bigosz.append(word)		
		if sim<0.30:
		   storan.append(word)
		if sim<0.20:
		   lillan.append(word)
	print "words below 0.50:"
	print len(bigosz)
	#print bigosz	
	print "words below 0.30:"
	print len(storan)
	#print storan
	print "words below 0.20:"
	print len(lillan)
	return sims
	#print lillan
	
def histogram(similarity_1,similarity_2):
	#title="Histogram of similarity between suffixed words in joint model"
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
	ax1.hist(similarity_1,bins=20)
	ax1.set_title("Skip-gram Joint German(pivots=1250)")
	ax1.set_ylabel("Number of words")
	ax1.set_xlabel("Cosine similarity")
	ax2.hist(similarity_2,bins=20)
	ax2.set_title("Skip-gram Joint German(pivots=100)")
	ax2.set_xlabel("Cosine similarity")
	#plt.title(title)
	plt.show()


#Pick a random number of words from the word_embeddings	
def pick_random_cluster(word_list,vecfile,sufflist,org_list):
	model=gensim.models.Word2Vec.load_word2vec_format(vecfile, binary=False)
	word=random.choice(word_list)
	selection_labels_suf=[]
	selection_labels_nosuf=[]
	vizlabels_suf=[]
	vizlabels_nosuf=[]
	if org_list!='none':
		embeddings=org_list
		countz=0
		for i in embeddings:
			if i+suffixes[0] in sufflist or i+suffixes[1] in sufflist:
				selection_labels_suf.append(i)
				vizlabels_suf.append(countz)
				countz+=1
			if i in sufflist:	
				selection_labels_nosuf.append(i)
				vizlabels_nosuf.append('black')
		selection_labels=selection_labels_suf+selection_labels_nosuf
		vizlabels=vizlabels_suf+vizlabels_nosuf			
	else:
		embeddings=model.most_similar(word,topn=2000)
		countz=0
		for i in embeddings:
			if  i[0]+suffixes[0] in sufflist or i[0]+suffixes[1] in sufflist:
				selection_labels_suf.append(i[0])
				vizlabels_suf.append(countz)
				countz+=1
			if i[0] in sufflist:
				selection_labels_nosuf.append(i[0])
				vizlabels_nosuf.append('black')
		selection_labels=selection_labels_suf[:10]+selection_labels_nosuf[:10]
		vizlabels=vizlabels_suf[:10]+vizlabels_nosuf[:10]
	return selection_labels,vizlabels		


	
	
def pick_word_cluster(word_list,vecfile,sufflist):
	model=gensim.models.Word2Vec.load_word2vec_format(vecfile, binary=False)
	#word=random.choice(word_list)
	selection_labels_suf=[]
	selection_labels_nosuf=[]
	vizlabels_suf=[]
	vizlabels_nosuf=[]
	#embeddings=model.most_similar(word,topn=2000)
	countz=0
	for i in word_list:
		if  i+suffixes[0] in sufflist or i+suffixes[1] in sufflist:
			selection_labels_suf.append(i)
			vizlabels_suf.append(countz)
			countz+=1
		if i in sufflist:
			selection_labels_nosuf.append(i)
			vizlabels_nosuf.append(countz)
			countz+=1
	selection_labels=selection_labels_suf[:100]+selection_labels_nosuf[:100]
	vizlabels=vizlabels_suf[:100]+vizlabels_nosuf[:100]
	return selection_labels,vizlabels

	
def checkup_wordpairs(transfrom_wordlist,transto_wordlist):
	word_pairs=[]
	for word in transfrom_wordlist:
		if word in transto_wordlist:
			word_pairs.append(word)
	return word_pairs		


def data_visulization_trans_rolf(selection_labels,transto,wordlist,suffixes,allwords,viz,crazy_words):
	dist_array=transto
	print dist_array.shape
	a = np.array(dist_array)
	pca = PCA(a)
	a2 = np.dot(a, pca.Wt[:2].T)
	#a = dist_array.astype(np.float64, copy=False)
	#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	#hej=tsne(a, 2, 100, 20.0)
	#print hej.shape
	selection_vector_transto=np.zeros((1,2))
	selection_vector_translation=np.zeros((1,2))
	selection_vector=np.zeros((1,2))
	suffz_1=[]
	for j,i in enumerate(selection_labels):
		if i+suffixes[0] in wordlist:
			suffz_1.append(j)
			num=wordlist.index(i+suffixes[0])
			selection_vector_transto=np.vstack((selection_vector_transto,a2[num]))
			selection_vector=np.vstack((selection_vector,a2[num]))	
	suffz_2=[]
	for j,i in enumerate(selection_labels):
		if i+suffixes[1] in wordlist:
			suffz_2.append(j)
			num=wordlist.index(i+suffixes[1])
			selection_vector_translation=np.vstack((selection_vector_translation,a2[(num)]))
			selection_vector=np.vstack((selection_vector,a2[num]))
	for i in selection_labels:
		if i in wordlist:
			num=wordlist.index(i)
			selection_vector=np.vstack((selection_vector,a2[num]))
		else:
			pass
	selection_vector=selection_vector[1:]
	nosuf=[]
	for j,i in enumerate(viz):
		if j not in suffz_1:
			if j not in suffz_2:
				nosuf.append(j)
	newviz=map(viz.__getitem__, suffz_1)+map(viz.__getitem__, suffz_2)+map(viz.__getitem__, nosuf)		
	selection_labels_crazy=[]
	for j,i in enumerate(selection_labels):
		if j in suffz_1:
			selection_labels_crazy.append((i+'_f'))
	for j,i in enumerate(selection_labels):
		if j in suffz_2:
			selection_labels_crazy.append((i+'_m'))
	for j,i in enumerate(selection_labels):
		if j not in suffz_1+suffz_2:
			selection_labels_crazy.append(i)		
	techno=len(viz)-viz.count('black')
	print selection_labels_crazy
	crazy_vector=np.zeros((1,2))
	very_crazy_words=[]
	for word in crazy_words:
		num=wordlist.index(word)
		crazy_vector=np.vstack((crazy_vector,a2[num]))
		word1=word.replace('_female', "_f")
		word2=word1.replace('_male', "_m")
		very_crazy_words.append(word2)
	print crazy_words	
	crazy_vector=crazy_vector[1:]
	for label, x, y in zip(very_crazy_words, crazy_vector[:,0], crazy_vector[:,1]):
		plt.text(x, y, label,color='gray',fontdict={'size': 8})
	plt.scatter(selection_vector[:,0], selection_vector[:,1], c='white')
	for sufn, label, x, y in zip(newviz, selection_labels_crazy, selection_vector[:,0], selection_vector[:,1]):
		if sufn=='black':
			plt.text(x, y, label,fontdict={'weight': 'bold', 'size': 11, 'color': 'black'})
		else:
			plt.text(x, y, label,color=plt.cm.Set1(sufn/techno),fontdict={'weight': 'bold','size': 11})	
	plt.xlim([-0.5,0.85])
	plt.ylim([0.4,-0.65])
	
	
	
def data_visulization_trans_rolf_vanilla(selection_labels,transto,wordlist,allwords,viz,crazy_words):
	dist_array=transto
	print dist_array.shape
	a = np.array(dist_array)
	pca = PCA(a)
	a2 = np.dot(a, pca.Wt[:2].T)
	#a = dist_array.astype(np.float64, copy=False)
	#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	#hej=tsne(a, 2, 100, 20.0)
	#print hej.shape
	selection_vector=np.zeros((1,2))
	for i in selection_labels:
		num=wordlist.index(i)
		selection_vector=np.vstack((selection_vector,a2[num]))
	selection_vector=selection_vector[1:]
	colz=['blue','green','red','cyan','magenta','orange','black','purple','gray','firebrick','chocolate','brown','hotpink','y','darkblue','sage','crimson','gold','plum','seagreen','indianred','lime','darkgreen','fuchsia']
	ax1.scatter(selection_vector[:,0], selection_vector[:,1], c='white')
	techno=len(viz)-viz.count('black')
	print techno
	crazy_vector=np.zeros((1,2))
	very_crazy_words=[]
	for word in crazy_words:
		num=wordlist.index(word)
		crazy_vector=np.vstack((crazy_vector,a2[num]))
		word.replace('_female', "_f")
		word.replace('_male', "_m")
		very_crazy_words.append(word)
	crazy_vector=crazy_vector[1:]
	for label, x, y in zip(very_crazy_words, crazy_vector[:,0], crazy_vector[:,1]):
		plt.text(x, y, label,color='gray',fontdict={'size': 8})
	for sufn, label, x, y in zip(viz, selection_labels, selection_vector[:,0], selection_vector[:,1]):
		if sufn=='black':
			plt.text(x, y, label,fontdict={'weight': 'bold', 'size': 11, 'color': 'black'})
		else:
			plt.text(x, y, label,color=plt.cm.Set1(sufn/techno),fontdict={'weight': 'bold', 'size': 13})
			#plt.text(x, y, label,fontdict={'weight': 'bold', 'size': 9, 'color': colz[sufn]})
	
	
vecfile_normal=sys.argv[3]
vecfile_suffixed_nogood=sys.argv[2]
vecfile_suffixed_good=sys.argv[1]
grouper=sys.argv[4]
if grouper=='age':
   suffixes=['_age_01','_age_03']
if grouper=='gender':
   suffixes=['_female','_male']

#Good stuff	
allwords=converter(vecfile_suffixed_good)
print len(allwords)
words_norm,dist_array_norm=converterX(vecfile_normal)
words_suff,dist_array_suff=converterX(vecfile_suffixed_good)
similarity_1=similarity_of_word_pairs(allwords,vecfile_suffixed_good)
word_list=[u'seri\xf6s',u'hochwertig',u'zu_handhaben',u'herzlichen_dank',u'begeistert',
u'meiner_meinung',u'vorbildlich',u'sch\xf6ner',u'beste',u'einen_guten',u'aufgrund',u'handhabung',u'guter_qualit\xe4t']
grejs=[u'vertrauensw\xfcrdig', u'flexibel', u'nachvollziehbar', u'stabil', u'gut_verarbeitet', u'ansprechend', u'qualitativ', u'attraktiv', u'zu_bedienen', u'intuitiv', u'aufgebaut', u'vielen_dank', u'besten_dank', u'dankesch\xf6n', u'danke', u'freunde', u'beeindruckt', u'total_begeistert', u'angenehm_\xfcberrascht', u'zufrieden', u'erstklassig', u'ausgezeichnet', u'hervorragend', u'\xfcberzeugend', u'optimal', u'spitze', u'viel_besser', u'absoluter', u'pers\xf6nlicher', u'toller', u'perfekter', u'aussehen', u'ausgefallen', u'g\xfcnstiger', u'dargestellt', u'original', u'qualitativ', u'kompetenter', u'schn\xe4ppchen', u'preiswerter', u'g\xfcnstigste', u'mit_abstand', u'schnellste', u'unschlagbar', u'solch', u'auf_grund', u'wegen', u'w\xe4hrend', u'anhand', u'bez\xfcglich', u'seitens', u'neben', u'faire_preise', u'guter_preis', u'g\xfcnstige_preise', u'problemlose_bestellung', u'einfache_bestellung', u'fairer_preis', u'gute_verpackung', u'einfacher_bestellvorgang', u'netter_kontakt', u'perfekter_service', u'perfekte_abwicklung', u'toller_service', u'unkomplizierte_bestellung', u'gute_qualit\xe4t', u'gutes_produkt', u'problemlose_abwicklung', u'gute_beratung', u'freundlicher_service', u'\xfcbersichtliche_seite', u'tolle_weine', u'einfache_bedienung', u'viel_auswahl', u'tolle_angebote', u'tolle_produkte', u'bedienung', u'anwendung', u'gestaltung', u'aufmachung', u'intuitiv', u'darstellung', u'bestellprozess', u'pr\xe4sentation', u'zusammenstellung', u'bestellvorgang', u'leicht_verst\xe4ndlich', u'anmeldung', u'versendung', u'bester_qualit\xe4t', u'kurzer_zeit', u'guten_zustand']	

extras=[]
poppo=[]
for word in grejs:
	if word in allwords:
		poppo.append(word)
	if word in words_suff:
		extras.append(word)
	for s in suffixes:
		if word+s in words_suff:
			extras.append(word+s)
			
			
model=gensim.models.Word2Vec.load_word2vec_format(sys.argv[1], binary=False)
relaz=[]
for word in word_list:
	if word in words_suff:
		embeddings=model.most_similar(word,topn=7)
		for i in embeddings:
			relaz.append(i[0])
	if word+suffixes[0] in words_suff: 	
		embeddings=model.most_similar(word+s,topn=20)
		for i in embeddings:
			relaz.append(i[0])
	if word+suffixes[1] in words_suff: 	
		embeddings=model.most_similar(word+s,topn=20)
		for i in embeddings:
			relaz.append(i[0])		
			

relaz2=[]
for i in relaz:
	if 'bildet_female' in i:
		pass	
	if 'ich_ges' in i:
		pass
	else:
		relaz2.append(i)		
	
selected_words,vizlabels=pick_word_cluster(word_list,vecfile_normal,words_suff)
#selected_words,vizlabels=pick_random_cluster(allwords,vecfile_normal,words_suff,'none')
print vizlabels
#f, (ax1, ax2) = plt.subplots(1, 2)
plt.suptitle("Skip-gram Joint model German (PCA)",fontsize=14)
plt.title("Pivots=1250, learning rate=0.025, word frequency cutoff=50",fontsize=10)
#data_visulization_trans_rolf_vanilla(selected_words,dist_array_norm,words_norm,allwords,vizlabels,poppo)
data_visulization_trans_rolf(selected_words,dist_array_suff,words_suff,suffixes,allwords,vizlabels,relaz2)
#ax2.set_title("Suffixed French model(pivots=1750)(PCA)")
#ettan = mpatches.Patch(color='black', label='pivots')
#plt.legend( handles=[ettan], loc = 'upper right', bbox_to_anchor = (0,-0.1,1,1), bbox_transform = plt.gcf().transFigure)
plt.show()
   
#First_word_list
words_norm,dist_array_norm=converterX(vecfile_normal)
words_suff_2,dist_array_suff_2=converterX(vecfile_suffixed_nogood)
allwords=converter(vecfile_suffixed_nogood)
similarity_2=similarity_of_word_pairs(allwords,vecfile_suffixed_nogood)
histogram(similarity_1,similarity_2)

selected_words_start,vizlabels=pick_random_cluster(allwords,vecfile_normal,words_suff_2,selected_words)
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Full French model(PCA)")
data_visulization_trans_rolf_vanilla(selected_words_start,dist_array_norm,words_norm,allwords,vizlabels)
data_visulization_trans_rolf(selected_words_start,dist_array_suff_2,words_suff_2,suffixes,allwords,vizlabels)
ax2.set_title("Suffixed French model(pivots=100)(PCA)")
ettan = mpatches.Patch(color='black', label='pivots')
plt.legend( handles=[ettan], loc = 'upper right', bbox_to_anchor = (0,-0.1,1,1), bbox_transform = plt.gcf().transFigure)
plt.show()	





#cosine distance
#Add titles
#Add legend
#Joint x-axis
#histogram(similarity_1,similarity_2)
allwords=converter(vecfile_suffixed_good)
words_norm,dist_array_norm=converterX(vecfile_normal)
words_suff,dist_array_suff=converterX(vecfile_suffixed_good)
months=[u'janvier',u'février',u'mars',u'avril',u'mai',u'juin',u'juillet',u'août',u'septembre',u'octobre',u'novembre',u'décembre']
selected_words,vizlabels=pick_random_cluster(allwords,vecfile_normal,words_suff,months)
print selected_words
print vizlabels
f, (ax1, ax2) = plt.subplots(1, 2)
data_visulization_trans_rolf_vanilla(selected_words,dist_array_norm,words_norm,allwords,vizlabels)
ax1.set_title("Full French model(PCA)")
data_visulization_trans_rolf(selected_words,dist_array_suff,words_suff,suffixes,allwords,vizlabels)
ax2.set_title("Suffixed French model(pivots=1750)(PCA)")
ettan = mpatches.Patch(color='black', label='pivots')
plt.legend( handles=[ettan], loc = 'upper right', bbox_to_anchor = (0,-0.1,1,1), bbox_transform = plt.gcf().transFigure)
plt.show()


"""
u'Paris',u'Lyon'
[u'Saint-Denis',u'Paris',u'Lyon',u'Toulouse',u'Marseille',u'Nice',u'Nantes',u'Montpellier',u'Lille',u'Reims'
u'Strasbourg',u'Bordeaux',u'Rennes',u'Reims',u'Le Havre',u'Saint-Étienne',u'Toulon',u'Grenoble',u'Dijon',
u'Angers',u'Villeurbanne']
"""











#Shared axis
#backman cluster
#show that cityparts should maybe not differ
#pick cluster 
#check range of colors




