#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
import matplotlib.pyplot as plt


### Task 1: Select what features you'll use.
# all features in dataset
initial_features_list =  ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', \
'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', \
'director_fees', 'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

# features i'll analyse
features_list = ['poi','salary', 'bonus', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees', 'other']

# features i created that are in final analysis
new_features_list = ['prop_stock_ex']

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

def data_overview(data_dictionary, features_list):
	num_data_points = len(data_dictionary)
	num_poi = 0
	missing_values = {}
	for feature in initial_features_list:
		missing_values[feature] = 0
	for elem in data_dictionary:
		if data_dictionary[elem]['poi'] == True:
			num_poi += 1
		for feature in data_dictionary[elem]:
			if data_dictionary[elem][feature] == 'NaN':
				missing_values[feature] += 1

	print '\nNumber of data points in the original dataset: ' + str(num_data_points)
	print '\nNumber of POIs in the dataset: ' + str(num_poi) + ' (' + str(int(float(num_poi)/num_data_points * 100)) + '%)'
	print '\nNumber of features used: ' + str(len(features_list))
	print '\nNumber of missing data points by feature: '
	print missing_values
	print ""


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
	initial_len_of_data_dict = len(data_dictionary)
	for feature in features_list:
		print 'Remove outliers from ' + feature + '...'
		people_to_remove = remove_feature_outliers(data_dictionary, feature)
		if len(people_to_remove) == 0:
			print 'No one to remove'
		for person in people_to_remove:
			print 'Removing ' + person
			del data_dictionary[person]
	final_len_of_data_dict = len(data_dictionary)
	print '\nAll outliers removed...' + str(final_len_of_data_dict) + ' observations remaining\n'

	return data_dictionary


### Task 3: Create new feature(s)
from sklearn.preprocessing import MinMaxScaler

def create_features(data_dictionary):
	for elem in data_dictionary:
		from_poi = float(data_dictionary[elem]['from_poi_to_this_person'])
		to_poi = float(data_dictionary[elem]['from_this_person_to_poi'])
		from_messages = float(data_dictionary[elem]['from_messages'])
		to_messages = float(data_dictionary[elem]['to_messages'])
		stock_ex = float(data_dictionary[elem]['exercised_stock_options'])
		stock_val = float(data_dictionary[elem]['total_stock_value'])
		bonus = float(float(data_dictionary[elem]['bonus']))
		total_payments = float(data_dictionary[elem]['total_payments'])
		deferred_income = float(data_dictionary[elem]['deferred_income'])
		shared_receipt = float(data_dictionary[elem]['shared_receipt_with_poi'])


		data_dictionary[elem]['prop_email_from_poi'] = from_poi / from_messages
		data_dictionary[elem]['prop_email_to_poi'] = to_poi / to_messages
		data_dictionary[elem]['prop_stock_ex'] = stock_ex / stock_val
		data_dictionary[elem]['prop_payments_as_bonus'] = bonus / total_payments
		data_dictionary[elem]['prop_income_not_deferred'] = deferred_income / total_payments
		data_dictionary[elem]['prop_messages_with_poi'] = (from_poi + to_poi + shared_receipt) / (from_messages + to_messages)


		for feature in new_features_list:
			try:
				type(int(data_dictionary[elem][feature])) == type(5)
			except:
				data_dictionary[elem][feature] = 'NaN'

	return data_dictionary




def create_new_features_list(features, new_features):
	for feature in new_features:
		features.append(feature)
	return features

def run(data_dict, features):
	# remove all outliers, return statistics on dataset and add new features
	new_features = list(features)
	new_features.remove('poi')
	appended_features = create_new_features_list(features, new_features_list)
	data_overview(data_dict, appended_features)
	data_dict = remove_all_outliers(data_dict, new_features)
	features = appended_features
	data_dict = create_features(data_dict)
	return data_dict, features


### Store to my_dataset for easy export below.
my_dataset, features_list = run(data_dict, features_list)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA

sss = StratifiedShuffleSplit(random_state = 42)

# decision tree parameters for the GridSearch
dec_tree_params = {}
dec_tree_params['dt__min_samples_split'] = [14, 15, 16]
dec_tree_params['dt__presort'] = [True, False]
dec_tree_params['dt__max_leaf_nodes'] = [None]
# dec_tree_params['dt__max_features'] = [2]
# dec_tree_params['dt__max_depth'] = [None, 8,6,4,2]
# dec_tree_params['dt__min_samples_leaf'] = [1,2,5,10]

# SVM parameters for the GridSearch
svm_params = {}
svm_params['svm__C'] = [1, 3, 5, 7, 9, 15]
svm_params['svm__gamma'] = [1, 3, 5, 7, 9, 15]
svm_params['svm__kernel'] = ['linear', 'rbf']

# Naive Bays parameters for the GridSearch
NB_params = {}

# AdaBoost parameters for GridSearch
Ada_params = {}
Ada_params['ada__base_estimator'] = [tree.DecisionTreeClassifier()]
Ada_params['ada__n_estimators'] = [15]
Ada_params['ada__learning_rate'] = [0.5, 1, 1.5]

params = {}
params['dt'] = dec_tree_params
#params['svm'] = svm_params
params['NB'] = NB_params
#params['ada'] = Ada_params

class_dict = {}
class_dict['dt'] = tree.DecisionTreeClassifier(random_state = 42)
class_dict['svm'] = SVC(random_state = 42)
class_dict['NB'] = GaussianNB()
class_dict['ada'] = AdaBoostClassifier(random_state = 42)

other_params = {}
other_params['kbest__k'] = [4]
#other_params['pca__n_components'] = [3, 4]
#other_params['pca__whiten'] = [True, False]


for dict in params:
	for parameter in other_params:
		params[dict][parameter] = other_params[parameter]

def classifier_grid(classifer_name):
	scaler = MinMaxScaler()
	classifier = class_dict[classifer_name]
	kbest = SelectKBest()
	pca = PCA(random_state = 42)
	gs = Pipeline(steps = [('scaling', scaler), ('kbest', kbest), (classifer_name, classifier)])
	gclf = GridSearchCV(gs, params[classifer_name], scoring = 'f1', cv = sss)
	gclf.fit(features, labels)
	clf = gclf.best_estimator_
	pred = clf.predict(features)
	accuracy = accuracy_score(pred, labels)
	recall = recall_score(pred, labels, average = 'weighted')
	precision = precision_score(pred,labels, average = 'weighted')
	f1 = f1_score(pred, labels, average = 'weighted')
	return clf, accuracy, recall, precision, f1

def tune_classifiers():
	result_dict = {}
	for param in params:
		best_est = classifier_grid(param)
		result_dict[param] = {}
		result_dict[param]['clf'] = best_est[0]
		result_dict[param]['accuracy'] = best_est[1]
		result_dict[param]['recall'] = best_est[2]
		result_dict[param]['precision'] = best_est[3]
		result_dict[param]['f1'] = best_est[4]

		print 'Classifier ' + param + ' has f1 score ' + str(result_dict[param]['f1'])
	print '\nSelecting best classifier based on f1 score...'
	return result_dict

results = tune_classifiers()

def best_classifier(results_dict):
	# runs through the Naive Bays, SVM and Decision tree and selects the one with the highest f1 score
	max_f1 = 0
	best_classifier = ''
	for entry in results_dict:
		if results_dict[entry]['f1'] > max_f1:
			max_f1 = results_dict[entry]['f1']
			best_classifier = results_dict[entry]['clf']

	return best_classifier, max_f1

best_classifier, max_f1 = best_classifier(results)

print '\n\n'

print 'optimal classifier selected...\n\nclassifier selected:\n'
print best_classifier
print '\n'
print 'f1 score: ' + str(max_f1)
print ""

final_features_list = []
support = best_classifier.named_steps['kbest'].get_support()
for i, boole in enumerate(support):
	if boole == True:
		final_features_list.append(features_list[i+1])

for classifier in ['dt', 'svm', 'NB', 'ada']:
	try:
		clf = best_classifier.named_steps[classifier]
		break
	except:
		pass

try:
	'Features used and relative importances:\n'
	feature_importance = clf.feature_importances_
	for i in range(0, len(feature_importance)):
		print final_features_list[i], feature_importance[i]
	print ""
except:
	pass

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
from tester import test_classifier
test_classifier(clf, data_dict, features_list)
