'''
10-fold validation of the decision tree
'''
from decision_tree import decision_tree_learning
from util import *
from copy import deepcopy

def prune_tree(data, tree):
    ''' prune one parent of 2 leaves if possible
        note that this function will always prune 1 parent node
        return: is_pruned, original tree, pruned tree'''
    labels = data[:,-1]
    original_tree_current_node = deepcopy(tree)
    current_node = tree

    # if current_node is a leaf
    if 'label' in current_node:
        return False, original_tree_current_node, current_node

    # if current_node is a parent of 2 unchecked leaves
    if 'label' in current_node['left'] and 'label' in current_node['right'] and current_node['left']['is_checked'] == False \
            and current_node['right']['is_checked'] == False:
        left_leaf = current_node['left']
        right_leaf = current_node['right']

        left_label = left_leaf['label']
        right_label = right_leaf['label']

        left_num = np.count_nonzero(labels == left_label)
        right_num = np.count_nonzero(labels == right_label)

        # change label according to the label with the
        #  maximum occurrence of label in the training dataset
        label = left_label if left_num > right_num else right_label

        # change current node to a leaf
        current_node = {'label': label, 'is_checked': False}

        # set original tree's children is_checked to True
        original_tree_current_node['left']['is_checked'] = True
        original_tree_current_node['right']['is_checked'] = True

        return True, original_tree_current_node, current_node

    # a parent with at least one child checked
    else:
        val = current_node['value']
        attribute = current_node['attribute']
        # prune left tree
        left_data = data[np.where(data[:,attribute] < val)]
        is_modified_left, original_tree_left, modified_tree_left = prune_tree( left_data , current_node['left'] )
        # if pruning is successful, return
        if is_modified_left == True:
            original_tree_current_node['left'] = original_tree_left
            current_node['left'] = modified_tree_left
            return True, original_tree_current_node, current_node

        # prune right tree
        right_data = data[np.where(data[:,attribute] >= val)]
        is_modified_right, original_tree_right, modified_tree_right = prune_tree( right_data , current_node['right'] )
        if is_modified_right == True :
            original_tree_current_node['right'] = original_tree_right
            current_node['right'] = modified_tree_right
            return True, original_tree_current_node, current_node

    # last case: if is a parent with 2 children checked
    return False, original_tree_current_node, current_node


def K_fold_pruning_evaluation(data, nr_of_folds = 10):
    ''' perform k-fold cross-validation on pruned tree and unpruned tree
    return: unpruned measure, pruned measure, a list of all (90) pruned trees'''
    #initiate arrays to store average measures across all folds
    all_folds_average_recall = []
    all_folds_average_precision = []
    all_folds_average_F1 = []
    all_folds_average_classification_rates = []

    pruned_all_folds_average_recall = []
    pruned_all_folds_average_precision = []
    pruned_all_folds_average_F1 = []
    pruned_all_folds_average_classification = []
    pruned_trees = []

    #shuffle data to avoid it being ordered by label
    np.random.shuffle(data)
    folds_split_1 = np.split(data, nr_of_folds)
    #loop trough all folds as the test data set
    for i in range(nr_of_folds):
        test_data_set = folds_split_1[i]
        training_validation_data_set =\
        np.concatenate(folds_split_1[0:i] + folds_split_1[i + 1:])
        ###################print status message#################
        print ('#'*70)
        print('USING FOLD {} AS THE TEST DATA'.format(i + 1))
        print('COMPARISON MEASURE: CLASSIFICATION RATE')
        print ('#'*70)
        training_validation_data_set_folds =\
        [index for index in range(nr_of_folds) if index != i]
        ########################################################
        #initiate arrays to store results for this fold
        recall_matrix = []
        precision_matrix = []
        F1_matrix = []
        classification_rates = []
        confusion_tensor = [] # patent pending by Olle

        pruned_recall_matrix = []
        pruned_precision_matrix = []
        pruned_F1_matrix = []
        pruned_classification_rates = []
        pruned_confusion_tensor = []

        #split up dataset
        folds_split_2 = np.split(training_validation_data_set, nr_of_folds - 1)
        #loop trough the remaining k-1 folds as the validation data set
        for index in range(nr_of_folds - 1):
            evaluation_data_set = folds_split_2[index]
            training_data_set =\
            np.concatenate(folds_split_2[0:index] + folds_split_2[index + 1:])
            ###################print status message#################
            print('WITH FOLD {} AS THE VALIDATION DATA'\
            .format(training_validation_data_set_folds[index]  + 1))
            print('TRAINING TREE ON REMAINING FOLDS...')
            ########################################################
            # train and evaluate the unpurned tree
            original_tree, _ = decision_tree_learning(training_data_set, 0)
            confusion_matrix , recall, precision, F1, classification_rate\
            = evaluate(evaluation_data_set, original_tree)
            ###################print status message#################
            print ("Tree depth:", get_depth(original_tree))
            print('The validation score for the trained tree: {}'\
            .format(classification_rate))
            ########################################################
            #keep a copy of the unpruned tree
            current_tree = deepcopy(original_tree)
            #prune
            print('PRUNING TREE...')
            while True:
                #flag returns flase if all possible prunes have been tested
                #and its not possible to prune anymore
                flag, current_tree, pruned_tree =\
                prune_tree(training_data_set, current_tree)
                #break if all nodes have been pruned
                if not flag:
                    break

                # evaluate pruned tree
                pruned_confusion_matrix , pruned_recall, pruned_precision, pruned_F1,\
                pruned_classification_rate = evaluate(evaluation_data_set, pruned_tree)
                #check if pruned tree is better
                if pruned_classification_rate >= classification_rate:
                    #if better update current tree and classification rate
                    current_tree = pruned_tree
                    classification_rate = pruned_classification_rate

            pruned_trees.append(current_tree)
            #evaluate unpruned and best pruned tree on validation dataset
            pruned_confusion_matrix , pruned_recall, pruned_precision, pruned_F1,\
            pruned_classification_rate = evaluate(evaluation_data_set, current_tree)
            ###################print status message#################
            print ("Pruned tree depth:", get_depth(current_tree) )
            print('The validation score for the best pruned tree: {}'\
            .format(pruned_classification_rate))
            print('TESTING TREES ON TEST DATA SET...')
            ########################################################
            #evaluate unpruned and best pruned tree on test dataset
            confusion_matrix , recall, precision, F1, classification_rate\
            = evaluate(test_data_set, original_tree)
            pruned_confusion_matrix, pruned_recall, pruned_precision, pruned_F1,\
            pruned_classification_rate = evaluate(test_data_set, current_tree)
            ###################print status message#################
            print('The test score for the original tree: {}'\
            .format(classification_rate))
            print('Confusion matrix for original tree:')
            print(confusion_matrix)
            print('The test score for the pruned tree: {}'\
            .format(pruned_classification_rate))
            print('Confusion matrix for pruned tree:')
            print(pruned_confusion_matrix)
            print ('-'*70)
            ########################################################
            #store measures
            classification_rates.append(classification_rate)
            pruned_classification_rates.append(pruned_classification_rate)
            #stack all label-wise measures as arrays(each fold is a row)
            #averages for each label are then the array columns averages(axis 0)
            if index == 0:
                recall_matrix = recall
                precision_matrix = precision
                F1_matrix = F1
                confusion_tensor = confusion_matrix

                pruned_recall_matrix = pruned_recall
                pruned_precision_matrix = pruned_precision
                pruned_F1_matrix = pruned_F1
                pruned_confusion_tensor = pruned_confusion_matrix
            else:
                recall_matrix = np.vstack((recall_matrix, recall))
                precision_matrix = np.vstack((precision_matrix, precision))
                F1_matrix = np.vstack((F1_matrix, F1))
                confusion_tensor = np.vstack((confusion_tensor, confusion_matrix))

                pruned_recall_matrix =\
                np.vstack((pruned_recall_matrix, pruned_recall))
                pruned_precision_matrix =\
                np.vstack((pruned_precision_matrix, pruned_precision))
                pruned_F1_matrix =\
                np.vstack((pruned_F1_matrix, pruned_F1))
                pruned_confusion_tensor =\
                np.vstack((pruned_confusion_tensor, pruned_confusion_matrix))

            print('-'*70)

        #calculate average of evaluation measures
        average_recall = np.mean(recall_matrix, axis=0)
        average_precision = np.mean(precision_matrix, axis=0)
        average_F1 = np.mean(F1_matrix, axis=0)
        average_classification_rate = np.mean(classification_rates)

        pruned_average_recall = np.mean(pruned_recall_matrix, axis=0)
        pruned_average_precision = np.mean(pruned_precision_matrix, axis=0)
        pruned_average_F1 = np.mean(pruned_F1_matrix, axis=0)
        pruned_average_classification_rate = np.mean(pruned_classification_rates)

        #store average measures across all folds
        all_folds_average_classification_rates\
        .append(average_classification_rate)
        pruned_all_folds_average_classification\
        .append(pruned_average_classification_rate)
        #stack all label-wise measures as arrays(each fold is a row)
        #averages for each label are then the array columns averages(axis 0)
        if i == 0:
            all_folds_average_recall = average_recall
            all_folds_average_precision = average_precision
            all_folds_average_F1 = average_F1

            pruned_all_folds_average_recall = pruned_average_recall
            pruned_all_folds_average_precision = pruned_average_precision
            pruned_all_folds_average_F1 = pruned_average_F1
        else:
            all_folds_average_recall =\
            np.vstack((all_folds_average_recall, average_recall))
            all_folds_average_precision =\
            np.vstack((all_folds_average_precision, average_precision))
            all_folds_average_F1 =\
            np.vstack((all_folds_average_F1, average_F1))

            pruned_all_folds_average_recall =\
            np.vstack((pruned_all_folds_average_recall, pruned_average_recall))
            pruned_all_folds_average_precision =\
            np.vstack((pruned_all_folds_average_precision, pruned_average_precision))
            pruned_all_folds_average_F1 =\
            np.vstack((pruned_all_folds_average_F1 , pruned_average_F1))


    #calculate average of evaluation measures across all folds
    average_recall =\
    np.mean(all_folds_average_recall, axis=0)
    average_precision =\
    np.mean(all_folds_average_precision, axis=0)
    average_F1 =\
    np.mean(all_folds_average_F1, axis=0)
    average_classification_rate =\
    np.mean(all_folds_average_classification_rates)

    pruned_average_recall =\
    np.mean(pruned_all_folds_average_recall, axis=0)
    pruned_average_precision =\
    np.mean(pruned_all_folds_average_precision, axis=0)
    pruned_average_F1 =\
    np.mean(pruned_all_folds_average_F1, axis=0)
    pruned_average_classification_rate =\
    np.mean(pruned_all_folds_average_classification)

    measures =\
    [average_classification_rate, average_recall, average_precision, average_F1]
    pruned_measures =\
    [pruned_average_classification_rate, pruned_average_recall,\
    pruned_average_precision, pruned_average_F1]
    improvment = pruned_average_classification_rate-average_classification_rate

    ###################print results#################
    print('Average test score for unpruned trees: {}'\
    .format(round(average_classification_rate, 3)))
    print('Average test score for pruned trees: {}'\
    .format(round(pruned_average_classification_rate, 3)))
    print('Pruning improved the average test score by {}%'\
    .format(round(improvment*100, 3)))
    print('Average recall for unpruned trees: {}'\
    .format(np.round(average_recall, 3)))
    print('Average precision for unpruned trees: {}'\
    .format(np.round(average_precision, 3)))
    print('Average F1 score for unpruned trees: {}'\
    .format(np.round(average_F1, 3)))
    print('Average recall for pruned trees: {}'\
    .format(np.round(pruned_average_recall, 3)))
    print('Average precision for pruned trees: {}'\
    .format(np.round(pruned_average_precision, 3)))
    print('Average F1 score for pruned trees: {}'\
    .format(np.round(pruned_average_F1, 3)))
    #################################################
    return measures, pruned_measures


def K_fold_evaluation(data, nr_of_folds = 10, shuffle = True):
    '''
    return: average confusion matrix, recall, precision, F1 score,
     classification rate across the K-folds.
    '''
    # shuffle the data so its not in labeled order
    if shuffle == True:
        np.random.shuffle(data)

    folds = np.split(data, nr_of_folds)

    # initiate arrays for storing evaluation measures
    recall_matrix = []
    precision_matrix = []
    F1_matrix = []
    classification_rates = []
    confusion_tensor = []
    trees = []

    for index in range(nr_of_folds):
        print('USING FOLD {} AS THE VALIDATION DATA'.format(index + 1))
        print('TRAINING TREE ON REMAINING FOLDS...')
        #pick out folds for training and testing
        test_data_set = folds[index]
        training_data_set = np.concatenate(folds[0:index] + folds[index + 1:])

        # train the tree
        tree, _ = decision_tree_learning(training_data_set, 0)
        trees.append(tree)
        #evaluate the tree
        confusion_matrix , recall, precision, F1, classification_rate\
        = evaluate(test_data_set, tree)
        print('-'*70)
        #store evaluation measures
        confusion_matrix = np.reshape(confusion_matrix,(1, 4, 4))
        classification_rates.append(classification_rate)

        if index == 0:
            recall_matrix = recall
            precision_matrix = precision
            F1_matrix = F1
            confusion_tensor = confusion_matrix
        else:
            recall_matrix = np.vstack((recall_matrix, recall))
            precision_matrix = np.vstack((precision_matrix, precision))
            F1_matrix = np.vstack((F1_matrix, F1))
            confusion_tensor = np.vstack((confusion_tensor, confusion_matrix))

    #calculate mean of evaluation measures
    average_recall = np.mean(recall_matrix, axis=0)
    average_precision = np.mean(precision_matrix, axis=0)
    average_F1 = np.mean(F1_matrix, axis=0)
    average_classification_rate = np.mean(classification_rates)
    average_confusion_matrix = np.mean(confusion_tensor, axis =0)

    return(average_classification_rate,
           average_recall,
           average_precision,
           average_F1,
           average_confusion_matrix,
           trees)


def evaluate(test_dataset, trained_tree):
    ''' return: confusion matrix, recall, precision, F1 score, classification rate'''
    confusion_matrix = np.zeros((4,4), dtype = int)

    # pass each sample in the trained_tree to get a predicted label
    for sample in test_dataset:
        current_node = trained_tree
        attribute_values = sample[:-1]
        true_label = int(sample[-1])

        while True:
            # if we find the leaf node
            # update confusion matrix and exit
            if 'label' in current_node:
                predicted_label = current_node['label']
                confusion_matrix[true_label -1, predicted_label - 1] += 1
                break
            # iterate through left node if attribute value is less than the split node value
            elif attribute_values[current_node['attribute']] < current_node['value']:
                current_node = current_node['left']
            # iterate through right node if attribute value is not less than the split node value
            else:
                current_node = current_node['right']

    recall, precision, F1, classification_rate = metrics(confusion_matrix)

    return confusion_matrix, recall, precision, F1, classification_rate


def metrics(confusion_matrix):
    ''' return: avg recall, avg precision, F1, classification rate'''
    num_classes = confusion_matrix.shape[0]
    recall = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    F1 = np.zeros(num_classes)

    for label in range(num_classes):
        # recall = true positive / sum of all predicted positive
        if np.sum(confusion_matrix[label,:]) != 0:
            recall[label] =\
            confusion_matrix[label,label]/np.sum(confusion_matrix[label,:])
        # precision is true positive / sum of actual positive
        if np.sum(confusion_matrix[:,label]) != 0:
            precision[label] =\
            confusion_matrix[label,label]/np.sum(confusion_matrix[:,label])
        if recall[label] != 0 and precision[label] != 0:
            F1[label] =\
            2*(precision[label]*recall[label])/(precision[label] + recall[label])

    classification_rate =\
    np.sum(np.diagonal(confusion_matrix))/np.sum(confusion_matrix)

    return recall, precision, F1, classification_rate


if __name__ == '__main__':
    pass
