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
features_list = ['poi','salary', 'bonus', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
def graph_feature_distribution(data_dict, feature):
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

def find_fence(data_dict, feature):
	feature_data = []
	for elem in data_dict:
		feature_data.append(data_dict[elem][feature])

	feature_numpy = np.asarray(feature_data).astype(float)
	feat_num_cl = feature_numpy[~np.isnan(feature_numpy)]
	q3, q1 = np.percentile(feat_num_cl, [75, 25])
	outer_max = 10 * (q3 - q1) + q3
	outer_min = q1 - 10 * (q3 - q1) 

	return outer_max, outer_min
	

upper, lower = find_fence(data_dict, 'salary')

def remove_feature_outliers(data_dict, feature):
	upper, lower = find_fence(data_dict, feature)
	data_dict_clean = {}

	for elem in data_dict:
		value = data_dict[elem][feature]
		if not (value != 'NaN' and (value > upper or value < lower)):
			data_dict_clean[elem] = data_dict[elem]

	return data_dict_clean



def remove_all_outliers(data_dict, features_list):
	for feature in features_list:
		data_dict_clean = remove_feature_outliers(data_dict, feature)
	return data_dict_clean



data_dict = remove_all_outliers(data_dict, features_list)




### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)