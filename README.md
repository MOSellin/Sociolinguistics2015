# Sociolinguistics2015
Thesis project

## this code runs in python 3.5

It requires the following liraries:
theano, tensorflow, sklearn, nltk, gensim, 
scipy, numpy, matplotlib, itertools, re,
operator,heapq,pickle

## run model from "Transformation Matrix paper" and save the results in a Pickle dictioary
python translationmatrix.py trained_vec_file(male) trained_vec_file(female) learning_rate    
Example: python translationmatrix.py C:\sociolinguistics_thesis\trained_files\standard_male_model_danish C:\sociolinguistics_thesis\trained_files\standard_female_model_danish 0.05 tm.p  

## run model from "Sogaard and Gouwls paper" and save the results in a Pickle dictioary
python pivotmodel.py corpora(female) corpora(male) dictionary_filename 
Example: python pivotmodel.py C:\sociolinguistics_thesis\corpora\denmark.F.train C:\sociolinguistics_thesis\corpora\denmark.M.train pivot.p

## run model from David bamman paper and save the results in a  dictionary
python bamman_model.py corpora(female) corpora(male) dictionary_filename learning_rate lamda_
Example: python bamman_model.py C:\sociolinguistics_thesis\corpora\denmark.F.train C:\sociolinguistics_thesis\corpora\denmark.M.train bamman.p 0.05 0.0000001
 
##Use models to perform a review classification task
python classifier.py dataset_path model_dictionary size gender
Example: python classifier2.py C:\sociolinguistics_thesis\corpora\ C:\sociolinguistics_thesis\trained_models_dict\bamman_danish.p gender 200

## Visualize results
python visual_evaluation.py model_dictionary size 
Example: python visual_evaluation.py C:\sociolinguistics_thesis\trained_models_dict\bamman_danish.p 200
