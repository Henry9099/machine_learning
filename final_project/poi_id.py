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
def graph_feature_distribution(feature):
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
	feature_data = []
	for elem in data_dict:
		feature_data.append(data_dict[elem][feature])

	feature_numpy = np.asarray(feature_data).astype(float)
	feat_num_cl = feature_numpy[~np.isnan(feature_numpy)]
	q3, q1 = np.percentile(feat_num_cl, [75, 25])
	outer_max = 10 * (q3 - q1) + q3
	outer_min = q1 - 10 * (q3 - q1) 

	return outer_max, outer_min
	

def remove_feature_outliers(feature):
	upper, lower = find_fence(feature)
	data_dict_clean = {}

	for elem in data_dict:
		value = data_dict[elem][feature]
		if not (value != 'NaN' and (value > upper or value < lower)):
			data_dict_clean[elem] = data_dict[elem]

	return data_dict_clean


def remove_all_outliers(features_list):
	for feature in features_list:
		data_dict_clean = remove_feature_outliers(feature)
	return data_dict_clean




### Task 3: Create new feature(s)
from sklearn.preprocessing import MinMaxScaler

def scale_feature(feature):
	value_list = []
	for elem in data_dict:
		value_list.append(data_dict[elem][feature])
	feature_numpy = np.asarray(value_list).astype(float)
	feat_num_cl = feature_numpy[~np.isnan(feature_numpy)]
	scaler = MinMaxScaler()
	rescaled_feature = scaler.fit_transform(feat_num_cl)

	i = 0
	for elem in data_dict:
		if data_dict[elem][feature] != 'NaN':
			data_dict[elem][feature] = rescaled_feature[i]
	return data_dict



def scale_all_features(features_list):
	for feature in features_list:
		data_dict = scale_feature(feature)
	return data_dict

def run(features):
	features.remove('poi')
	data_dict = remove_all_outliers(features)
	data_dict = scale_all_features(features)

	return data_dict


data_dict = run(features_list)
'''
i = 0
for elem in data_dict:
	while i <1:
		print data_dict[elem]
		i +=1
'''

### Store to my_dataset for easy export below.
my_dataset = data_dict



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)


for entry in data:
	if entry[0] == 0:
		entry[0] = True
	else:
		entry[0] = False



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
	print 'accuracy of Naive Bayes is: ' + str(accuracy)

if False:
	clf = SVC(kernel = 'linear')
	clf.fit(features, labels)
	pred = clf.predict(features)
	accuracy = accuracy_score(pred, labels)
	print 'accuracy of SVM is: ' + str(accuracy)

if False:
	clf = tree.DecisionTreeClassifier(min_samples_split = 5)
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)
	accuracy = accuracy_score(pred, labels_test)
	print 'accuracy of Decision Tree is: ' + str(accuracy)

def run_classifier(classifier, name, features_train, features_test, labels_train, labels_test):
	clf = classifier
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)
	accuracy = accuracy_score(pred, labels_test)
	recall = recall_score(pred, labels_test)
	precision = precision_score(pred, labels_test)
	print 'accuracy of ' + name + ' : ' +str(accuracy)
	print 'recall of ' + name + ' : ' +str(recall)
	print 'precision of ' + name + ' : ' +str(precision)


classifiers = {'Naive Bays': [GaussianNB(), 'Naive Bays'],\
				'SVM': [SVC(kernel = 'rbf', gamma = 10), 'SVM'],\
				'Decision Tree': [tree.DecisionTreeClassifier(min_samples_split = 100), 'Decision Tree']}


def run_all_classifiers(features_train, features_test, labels_train, labels_test):
	for classifier in classifiers:
		#print classifiers[classifier][0]
		run_classifier(classifiers[classifier][0], classifiers[classifier][1], features_train, features_test, labels_train, labels_test)


if True:
	run_all_classifiers(features_train, features_test, labels_train, labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)