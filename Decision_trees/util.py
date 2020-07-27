import numpy as np
import matplotlib.pyplot as plt

def import_clean_data():
	return np.loadtxt('./dataset/clean_dataset.txt')

def import_noisy_data():
	return np.loadtxt('./dataset/noisy_dataset.txt')

def import_new_data(filename):
	return np.loadtxt(filename)

def get_depth(tree):
	if 'label' in tree :
		return 0
	return (max(get_depth(tree['right']), get_depth(tree['left']) ) + 1)
