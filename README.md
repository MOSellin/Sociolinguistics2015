# Sociolinguistics2015
Thesis project

## this code runs in python 3.5

It requires the following libraries:
theano, tensorflow, sklearn, nltk, gensim, 
scipy, numpy, matplotlib, itertools, re,
operator,heapq,pickle

Machine learning within Natural Language Processing (NLP) models language as uniform. While this gives good results for several NLP-tasks, sociolinguistics have shown that within a language, groups use different words to express feelings or concepts. Factors such as age, ethnicity and location are known to influence the use of language. However, sociolinguistic studies primarily use qualitative analysis, and only recently have differences in language use been explored using statistics based on corpora. Vector space models are learned from distributional information in a corpus. Each word is represented as a word embedding, which is a vector of numbers holding a relative position to the rest of the vocabulary. The aim of this thesis is to add sociolinguistic factors to vector space models and evaluate if the word embeddings can improve existing NLP-tasks. We evaluate three approaches of using author information in Skip-gram vector space models, and use gender and age as sociolinguistic factors.

Distributed representations of geographically situated language.(David Bamman, Chris Dyer, and A. Noah Smith, https://aclanthology.org/P14-2134.pdf), have extended the original Skip-Gram model, by adding additional embeddings depending on the context of language use. Where Skip-gram word embeddings vwi are represented in one matrix V, the implementation (S-G-Bamman) adds an additional number of matrices. These matrices will hold word embeddings which represents a value dependent of the context of language use.In this thesis, we use j sociolinguistic groups to represent contexts of language use in the matrices.
 
Our intuition is that the most frequent words in a language do not vary in use due to sociolinguistic factors. Common words such as denominators, numbers and prepositions are most likely used in the same way within one language. To model this, we use a Skip-gram Joint Model. We assume that the p most frequent words in a corpus, called pivots, have the same syntactical meaning for all sociolinguistic groups. For all but the p most frequent words, we initialize separate word embeddings. A similar technique is used in the paper Simple task-specific bilingual word embeddings by Stephan Gouws and
Anders Søgaard. (https://aclanthology.org/N15-1157.pdf)


As in Exploiting Similarities among Languages for Machine Translation by Tomas Mikolov, Quoc V. Le and Ilya Sutskever, (https://arxiv.org/pdf/1309.4168.pdf), two separately trained Skip-gram vector spaces in different languages can be mapped linearly by using a transformation matrix, which we refer to as Translation Matrix TM. In this thesis, we instead use two sociolinguistic groups in one language.
If we train separate Skip-gram models for two sociolinguistic groups, we can represent them
as two vector spaces X and Z. We may then use a set of words which occur in both models to form word pairs and match the words embeddings in both vector spaces as xi and zi.

As extrinsic evaluation, we perform a topic classification task. As an intrinsic evaluation, we
use dimensionality-reduced visualization for samples of word embeddings. For evaluation, we use 0.8 million product and service reviews taken
from Trustpilot.com.

Our results show a joint model with separately trained word embeddings for less frequent words performs similarly to both uniform language baselines. Intrinsic evaluation shows separately trained word embeddings form distinct clusters, but also shows the selection of group-separated words could be improved. A more refined selection of words, along with more precise parameter tuning could represent sociolinguistic factors better, and possibly improve NLP-tasks. (https://arxiv.org/pdf/1309.4168.pdf).


## run model from "Exploiting Similarities among Languages for Machine Translation" and save the results in a Pickle dictioary
python translationmatrix.py trained_vec_file(male) trained_vec_file(female) learning_rate    
Example: python translationmatrix.py Sociolinguistics2015/trained_files/standard_male_model_danish Sociolinguistics2015/trained_files/standard_female_model_danish 0.05 tm.p  

## run model from "Sogaard and Gouwls paper" and save the results in a Pickle dictioary
python pivotmodel.py corpora(female) corpora(male) dictionary_filename 
Example: python pivotmodel.py C:\sociolinguistics_thesis\corpora\denmark.F.train C:\sociolinguistics_thesis\corpora\denmark.M.train pivot.p

## run model from David Bamman paper and save the results in a  dictionary
Sociolinguistics2015
python bamman_model.py corpora(female) corpora(male) dictionary_filename learning_rate lamda_
Example: python bamman_model.py Sociolinguistics2015/corpora/denmark.F.train denmark.M.train bamman.p 0.05 0.0000001
 
##Use models to perform a review classification task
python classifier.py dataset_path model_dictionary size gender
Example: python classifier2.py Sociolinguistics2015/corpora Sociolinguistics2015/corpora/trained_models_dict/bamman_danish.p M 200

## Visualize results
python visual_evaluation.py model_dictionary size 
Example: python visual_evaluation.py Sociolinguistics2015/trained_models_dict\bamman_danish.p 200
