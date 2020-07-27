from evaluation import *

def STEP3():
    """
    This is running the code for the STEP 3 fo the corsework spec which is
    evaluation of the the accuracy using 10-fold cross validation on both
    the clean and noisy datasets.
    :retuns:
    """
    clean_data = import_clean_data()
    print ('#'*70)
    print('10-FOLD CROSSVALIDATION ON THE CLEAN DATA')
    print ('#'*70)
    (average_classification_rate,
    average_recall,
    average_precision,
    average_F1,
    average_confusion_matrix,
    trees) =\
    K_fold_evaluation(clean_data)
    print('DECISION TREE RESULTS ON CLEAN DATA')
    print('Average confusion matrix for clean data:')
    print(np.round(average_confusion_matrix, 0))
    print('Average recall per class for clean data: {}'\
    .format(np.round(average_recall, 3)))
    print('Average precision per class for clean data: {}'\
    .format(np.round(average_precision, 3)))
    print('Average F1-score per class for clean data: {}'\
    .format(np.round(average_F1, 3)))
    print('Average classification rate for clean data: {}'\
    .format(np.round(average_classification_rate, 3)))

    noisy_data = import_noisy_data()
    print ('#'*70)
    print('10-FOLD CROSSVALIDATION ON THE NOISY DATA')
    print ('#'*70)
    (average_classification_rate,
    average_recall,
    average_precision,
    average_F1,
    average_confusion_matrix,
    trees) =\
    K_fold_evaluation(noisy_data)
    print('DECISION TREE RESULTS ON NOISY DATA')
    print('Average confusion matrix for noisy data:')
    print(np.round(average_confusion_matrix, 0))
    print('Average recall per class for noisy data: {}'\
    .format(np.round(average_recall, 3)))
    print('Average presicion per class for noisy data: {}'\
    .format(np.round(average_precision, 3)))
    print('Average F1-score per class for noisy data: {}'\
    .format(np.round(average_F1, 3)))
    print('Average classification rate for noisy data: {}'\
    .format(np.round(average_classification_rate, 3)))


def STEP4():
    clean_data = import_clean_data()
    print ('#'*70)
    print('PRUNING AND 10-FOLD CROSSVALIDATION ON THE CLEAN DATA')
    print ('#'*70)
    K_fold_pruning_evaluation(clean_data)

    noisy_data = import_noisy_data()
    print ('#'*70)
    print('PRUNING AND 10-FOLD CROSSVALIDATION ON THE NOISY DATA')
    print ('#'*70)
    K_fold_pruning_evaluation(noisy_data)

if __name__ == '__main__':
    STEP3()
    STEP4()
