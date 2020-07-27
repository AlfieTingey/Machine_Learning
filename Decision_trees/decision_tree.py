from util import *

def H(labels):
	''' return H-function of labels'''
	total = len(labels)
	_, counts = np.unique(labels, return_counts=True)
	probs = counts / total

	return -sum(probs * np.log2(probs))

def gain(dataset, left_data, right_data):
	''' return gain value of a attribute according to left and right data'''
	len_all = len(dataset)
	len_left = len(left_data)
	len_right = len(right_data)

	remainder = (H(left_data)*len_left + H(right_data)*len_right) / len_all

	return H(dataset) - remainder

def attribute_gain(dataset, att_index):
	''' return: (max gain, value) to split an index-th attribute
		according to its gain value'''
	# iterate through each value in attribute and compute gain
	best_attribute_split = (-float('inf'), 0, [], []) # best attribute gain, best value, left_data, right_data

	sorted_data = dataset[dataset[:,att_index].argsort()] # sort data by attribute
	attribute_values = np.unique(dataset[:,att_index])

	# split data to left and right, and find the best att_index to split
	for val in attribute_values:
		left_data = sorted_data[np.where(sorted_data[:,att_index] < val)]
		right_data = sorted_data[np.where(sorted_data[:,att_index] >= val)]

		attribute_gain = gain(sorted_data, left_data, right_data)

		if attribute_gain > best_attribute_split[0]:
			best_attribute_split = (attribute_gain, val, left_data, right_data)

	return best_attribute_split

def find_split(dataset):
	''' return: best attribute, value, left data and right data for the best split'''
	best_gain = -float('inf')
	best_split = (0, 0, [], []) # best attribute, best value, left_data, right_data

	# loop all attribute to find the best attribute to split
	for att_i in range(len(dataset[0])-1):
		gain, value, left_data, right_data = attribute_gain(dataset, att_i)
		if gain > best_gain:
			best_gain = gain
			best_split = (att_i, value, left_data, right_data)

	return best_split

def decision_tree_learning(dataset, depth=0):
	''' return: root node of the decision tree '''
	# check if all values in the label are the same
	if len(np.unique(dataset[:,-1])) == 1:
		return{'label': int(dataset[:,-1][0]), 'is_checked' : False}, depth

	else:
		# find attribute and value to split the dataset
		attribute, value, left_data, right_data = find_split(dataset)

		# recursively calls the fucntion on left and right node
		left_node, left_depth = decision_tree_learning(left_data, depth+1)
		right_node, right_depth = decision_tree_learning(right_data, depth+1)

		root = {'attribute': attribute, 'value': value, 'left': left_node, 'right': right_node}

		return root, max(left_depth, right_depth)

