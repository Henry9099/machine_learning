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
features_list = ['poi','salary', 'bonus', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock']
new_features = ['prop_email_from_poi', 'prop_stock_exercised']
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
		print 'length of data_dict before removing ' + feature + ' : ' + str(len(data_dict))
		people_to_remove = remove_feature_outliers(data_dictionary, feature)
		for person in people_to_remove:
			del data_dictionary[person]
		print 'length of data_dict after removing ' + feature + ' : ' + str(len(data_dict))

	return data_dictionary




### Task 3: Create new feature(s)
from sklearn.preprocessing import MinMaxScaler

def create_features(data_dictionary):
	for elem in data_dictionary:
		# add proportion of from_emails from poi
		try:
			data_dictionary[elem]['prop_email_from_poi'] = int(data_dictionary[elem]['from_poi_to_this_person']) / int(data_dictionary[elem]['from_messages'])
		except:
			data_dictionary[elem]['prop_email_from_poi'] = 'NaN'

		# add exercised_stock_options as proportion of total_stock_value
		try:
			data_dictionary[elem]['prop_stock_exercised'] = int(data_dictionary[elem]['exercised_stock_options']) / int(data_dictionary[elem]['total_stock_value'])
		except:
			data_dictionary[elem]['prop_stock_exercised'] = 'NaN'



	return data_dictionary

'''
for elem in data_dict:
	if data_dict[elem]['bonus'] != 'NaN':
		print elem, data_dict[elem]['poi']
'''

def min_max(feature):
	#find minimum and maximum value of feature, ignoring NaNs - now deprecated as done in the Pipline
	value_list = []
	for elem in data_dict:
		value_list.append(data_dict[elem][feature])
	feature_numpy = np.asarray(value_list).astype(float)
	maxi = np.nanmax(feature_numpy)
	mini = np.nanmin(feature_numpy)
	return mini, maxi


def scale_feature(feature):
	# manual MinMax scalar as to avoid NaNs - now deprecated as done in the Pipline
	mini, maxi = min_max(feature)
	for elem in data_dict:
		value = data_dict[elem][feature]
		if value != 'NaN':
			data_dict[elem][feature] = (data_dict[elem][feature] - mini) / (maxi - mini)
	return data_dict

def scale_all_features(features_list):
	#loop through all features and scale - now deprecated as done in the Pipline
	for feature in features_list:
		data_dict = scale_feature(feature)
	return data_dict

def create_new_features_list(features, new_features):
	for feature in new_features:
		features.append(feature)
	return features

def run(data_dict, features):
	print 'initial length of data_dict: ' + str(len(data_dict))
	# remove all outliers and scale all features except our target 'poi'
	new_features = list(features)
	new_features.remove('poi')
	data_dict = remove_all_outliers(data_dict, new_features)
	print 'length of data_dict after removing outliers: ' + str(len(data_dict))
	#data_dict = scale_all_features(new_features)
	features = create_new_features_list(features, new_features)
	print features 
	data_dict = create_features(data_dict)

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

# deprecated due to classifierGrid below
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


# deprecated due to classifierGrid below
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

# deprecated due to classifierGrid below
if False:
	clf = tree.DecisionTreeClassifier(min_samples_split = 10)
	clf.fit(features, labels)
	pred = clf.predict(features)
	accuracy = accuracy_score(pred, labels)
	recall = recall_score(pred, labels)
	precision = precision_score(pred, labels)
	print clf
	'''print 'accuracy of Decision Trees is: ' + str(accuracy)
	print 'recall of Decision Trees is: ' + str(recall)
	print 'precision of Decision Trees is: ' + str(precision)'''



from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest

sss = StratifiedShuffleSplit(random_state = 42)

# decision tree parameters for the GridSearch
dec_tree_params = {}
dec_tree_params['dt__min_samples_split'] = [5,10,15,20]
dec_tree_params['dt__presort'] = [True, False]
#dec_tree_params['dt__max_features'] = [2]
#dec_tree_params['dt__max_depth'] = [None, 8,6,4,2]
#dec_tree_params['dt__min_samples_leaf'] = [1,2,5,10]
#dec_tree_params['dt__max_leaf_nodes'] = [None, 4,8,12,20]

# SVM parameters for the GridSearch
svm_params = {}
svm_params['svm__C'] = [1,3,5,7,9,15]
svm_params['svm__gamma'] = [1,3,5,7,9,15]
svm_params['svm__kernel'] = ['linear', 'rbf']

# Naive Bays parameters for the GridSearch
NB_params = {}

params = {}
params['dt'] = dec_tree_params
params['svm'] = svm_params
params['NB'] = NB_params

class_dict = {}
class_dict['dt'] = tree.DecisionTreeClassifier()
class_dict['svm'] = SVC()
class_dict['NB'] = GaussianNB()

other_params = {}
other_params['kbest__k'] = [3]


for dict in params:
	for parameter in other_params:
		params[dict][parameter] = other_params[parameter]


def classifier_grid(classifer_name):
	scaler = MinMaxScaler()
	classifier = class_dict[classifer_name]
	kbest = SelectKBest(k=2)
	gs = Pipeline(steps = [('scaling', scaler),('kbest', kbest), (classifer_name, classifier)])
	gclf = GridSearchCV(gs, params[classifer_name], scoring = 'f1', cv = sss)
	gclf.fit(features, labels)
	clf = gclf.best_estimator_
	pred = clf.predict(features)
	accuracy = accuracy_score(pred, labels)
	recall = recall_score(pred, labels, average = 'weighted')
	precision = precision_score(pred,labels, average = 'weighted')
	return clf, accuracy, recall, precision

def tune_classifiers():
	result_dict = {}
	for param in params:
		best_est = classifier_grid(param)
		result_dict[param] = {}
		result_dict[param]['clf'] = best_est[0]
		result_dict[param]['accuracy'] = best_est[1]
		result_dict[param]['recall'] = best_est[2]
		result_dict[param]['precision'] = best_est[3]
	return result_dict

results = tune_classifiers()

def show_results(results_dict):
	for entry in results_dict:
		print ""
		print entry
		print 'accuracy: ' + str(results_dict[entry]['accuracy'])
		print 'recall: ' + str(results_dict[entry]['recall'])
		print 'precision: ' + str(results_dict[entry]['precision'])

show_results(results)

clf = results['dt']['clf']
clf = clf.named_steps['dt']
print clf
feature_importance = clf.feature_importances_
print features_list[1:]
print feature_importance


'''
scaler = MinMaxScaler()
dt = tree.DecisionTreeClassifier()
gs = Pipeline(steps = [('scaling', scaler), ('dt', dt)])
dtcclf = GridSearchCV(gs, dec_tree_params, scoring = 'f1', verbose = 10)
dtcclf.fit(features,labels)
clf = dtcclf.best_estimator_
print clf

pred = clf.predict(features)
accuracy = accuracy_score(pred, labels)
print accuracy
'''





if False:
	run_all_classifiers(features_train, features_test, labels_train, labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

if False:
	dump_classifier_and_data(clf, my_dataset, features_list)
