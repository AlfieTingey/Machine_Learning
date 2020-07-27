from evaluation import *
from util import import_new_data
import sys

new_data_file_name = sys.argv[1]
new_data = import_new_data(new_data_file_name)


# this block will run the 10-fold crossvalidation with pruning on
#the data imported and print all results as it as it goes and average results at the end
print ('#'*70)
print('PRUNING AND 10-FOLD CROSSVALIDATION ON UNSEEN DATA')
print ('#'*70)
K_fold_pruning_evaluation(new_data)
