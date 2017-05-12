#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
import matplotlib.pyplot as plt


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)

### Task 2: Remove outliers
def graph_feature_distribution(data_dict, feature):
	# create a histogram of the feature's distribution
	feature_data = []
	for elem in data_dict:
		feature_data.append(data_dict[elem][feature])

	feature_numpy = np.asarray(feature_data).astype(float)
	feat_num_cl = feature_numpy[~np.isnan(feature_numpy)]
	maxi = max(feat_num_cl)
	mini = min(feat_num_cl)
	binwidth = (maxi-mini)/100
	plt.hist(feat_num_cl, bins = np.arange(min(feat_num_cl), max(feat_num_cl) + binwidth, binwidth))
	plt.xlabel(feature)
	plt.ylabel('count')
	plt.show()

def find_fence(feature):
	# using a very wide fence as i only intend to catch extreme outliers/ wrong data entries
	feature_data = []
	for elem in data_dict:
		feature_data.append(data_dict[elem][feature])

	feature_numpy = np.asarray(feature_data).astype(float)
	feat_num_cl = feature_numpy[~np.isnan(feature_numpy)]
	q3, q1 = np.percentile(feat_num_cl, [75, 25])
	outer_max = 10 * (q3 - q1) + q3
	outer_min = q1 - 10 * (q3 - q1) 

	return outer_max, outer_min
	

def remove_feature_outliers(data_dict, feature):
	# remove the outliers for a particular feature
	upper, lower = find_fence(feature)
	data_dict_clean = {}
	elem_to_remove = []

	for elem in data_dict:
		value = data_dict[elem][feature]
		if  value == 'NaN' or (value < upper and value > lower):
			data_dict_clean[elem] = data_dict[elem]
		else:
			elem_to_remove.append(elem)
	return elem_to_remove


def remove_all_outliers(data_dictionary, features_list):
	# loop through all features in feature list and remove outliers
	for feature in features_list:
		people_to_remove = remove_feature_outliers(data_dictionary, feature)
		for person in people_to_remove:
			del data_dictionary[person]

	return data_dictionary




### Task 3: Create new feature(s)
from sklearn.preprocessing import MinMaxScaler

def min_max(feature):
	#find minimum and maximum value of feature, ignoring NaNs
	value_list = []
	for elem in data_dict:
		value_list.append(data_dict[elem][feature])
	feature_numpy = np.asarray(value_list).astype(float)
	maxi = np.nanmax(feature_numpy)
	mini = np.nanmin(feature_numpy)
	return mini, maxi


def scale_feature(feature):
	# manual MinMax scalar as to avoid NaNs
	mini, maxi = min_max(feature)
	for elem in data_dict:
		value = data_dict[elem][feature]
		if value != 'NaN':
			data_dict[elem][feature] = (data_dict[elem][feature] - mini) / (maxi - mini)
	return data_dict

def scale_all_features(features_list):
	#loop through all features and scale
	for feature in features_list:
		data_dict = scale_feature(feature)
	return data_dict

def run(data_dict, features):
	# remove all outliers and scale all features except our target 'poi'
	new_features = list(features)
	new_features.remove('poi')
	data_dict = remove_all_outliers(data_dict, new_features)
	#data_dict = scale_all_features(new_features)

	return data_dict


### Store to my_dataset for easy export below.
my_dataset = run(data_dict, features_list)




### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)


labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.3, random_state=42)


if False:
	clf = GaussianNB()
	clf.fit(features,labels)
	pred = clf.predict(features)
	accuracy = accuracy_score(pred, labels)
	recall = recall_score(pred, labels)
	precision = precision_score(pred, labels)
	print 'accuracy of Naive Bayes is: ' + str(accuracy)
	print 'recall of Naive Bayes is: ' + str(recall)
	print 'precision of Naive Bayes is: ' + str(precision)

if False:
	clf = SVC(kernel = 'rbf', gamma = 5, C = 5)
	clf.fit(features,labels)
	pred = clf.predict(features)
	accuracy = accuracy_score(pred, labels)
	recall = recall_score(pred, labels)
	precision = precision_score(pred, labels)
	print 'accuracy of SVM is: ' + str(accuracy)
	print 'recall of SVM is: ' + str(recall)
	print 'precision of SVM is: ' + str(precision)

if False:
	clf = tree.DecisionTreeClassifier(min_samples_split = 10)
	clf.fit(features, labels)
	pred = clf.predict(features)
	accuracy = accuracy_score(pred, labels)
	recall = recall_score(pred, labels)
	precision = precision_score(pred, labels)
	print 'accuracy of Decision Trees is: ' + str(accuracy)
	print 'recall of Decision Trees is: ' + str(recall)
	print 'precision of Decision Trees is: ' + str(precision)


def run_classifier(classifier, name, features_train, features_test, labels_train, labels_test):
	clf = classifier
	clf.fit(features, labels)
	pred = clf.predict(features)
	accuracy = accuracy_score(pred, labels)
	recall = recall_score(pred, labels)
	precision = precision_score(pred, labels)
	print '\n'	
	print 'accuracy of ' + name + ' : ' +str(accuracy)
	print 'recall of ' + name + ' : ' +str(recall)
	print 'precision of ' + name + ' : ' +str(precision)



classifiers = {'Naive Bays': [GaussianNB(), 'Naive Bays'],\
				'SVM': [SVC(kernel = 'rbf', gamma = 5, C = 5), 'SVM'],\
				'Decision Tree': [tree.DecisionTreeClassifier(min_samples_split = 10), 'Decision Tree']}


dec_tree_params = {}
dec_tree_params['min_samples_split'] = [5,10,15,20]
dec_tree_params['presort'] = [True, False]

svm_params = {}
svm_params['kernel'] = ['linear', 'rbf']
svm_params['C'] = [1,3,5,7,9,15]



print dec_tree_params

def run_all_classifiers(features_train, features_test, labels_train, labels_test):
	for classifier in classifiers:
		run_classifier(classifiers[classifier][0], classifiers[classifier][1], features_train, features_test, labels_train, labels_test)


if True:
	run_all_classifiers(features_train, features_test, labels_train, labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)


