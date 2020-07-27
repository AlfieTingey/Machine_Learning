import numpy as np
import pickle
import pandas as pd

from keras import Sequential, Input, Model
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.models import load_model as load_keras_model
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc

import matplotlib.pyplot as plt

N_FOLDS = 4

def create_model(num_hidden_layers=3, num_hidden_neurons=60, dropout=0.2, lr=0.001, weight_init='glorot_normal', optimizer='Adam'):
    model = Sequential()

    model.add(Dense(30, input_dim=9, activation='relu', kernel_initializer=weight_init))
    model.add(Dropout(0.2))
    for _ in range(num_hidden_layers):
        model.add(Dense(num_hidden_neurons, activation='relu', kernel_initializer=weight_init))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=weight_init))

	# Compile model
    if optimizer == 'RMSProp':
        model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

    return model


class ClaimClassifier:
    def __init__(self, model=None, num_hidden_layers=3, num_hidden_neurons=60, dropout=0.2, lr=0.001, weight_init='glorot_normal', optimizer='Adam', num_epochs=20, batch_size=20):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        # if initialize instance with saved model i.e. keras 'model.h5'
        if model:
            self.model = model
        else:
            self.model = create_model(num_hidden_layers, num_hidden_neurons, dropout, lr, weight_init, optimizer)
            
        # print model summary
        self.model.summary()

        self.num_epochs = num_epochs
        self.batch_size = batch_size
    

    # reset function used for evaluation when needed to create new model instance
    def reset(self):
        self.model = create_model()

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        X: numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
            A clean data set that is used for training and prediction.
        """

        # normalize to mean 0 and variance 1
        return pd.DataFrame(preprocessing.scale(X_raw))

    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded
        y_raw : numpy.ndarray (optional)
            A one dimensional numpy array, this is the binary target variable

        Returns
        -------
        ?
        """ 
        # create dataframe of y with column name 'made_claim'
        y_df = pd.DataFrame(y_raw)
        y_df.columns = ['made_claim']

        # normalize data 
        X_clean = self._preprocessor(X_raw)

        # remove rows to deal with imbalanced class
        # randomly remove rows with label 0 to match the number of rows with label 1
        num_rows = len(y_raw)
        num_class_one = len(y_df.query('made_claim != 0'))

        to_drop_indices = y_df.query('made_claim == 0').sample(num_rows - 2*num_class_one).index
        X_train = X_clean.drop(to_drop_indices).reset_index(drop=True)
        y_train = y_df.drop(to_drop_indices).reset_index(drop=True)

        # train model
        history = self.model.fit(X_train, y_train, epochs=self.num_epochs, batch_size=self.batch_size, shuffle=True, verbose=1)

        # print accuracy
        train_loss, train_acc = self.model.evaluate(X_train, y_train)
        print('Train loss: {:.3f} Train accuracy: {:.3f}'.format(train_loss, train_acc))

        ########### plot history ###############
        # plt.clf()
        # ax1 = plt.subplot(2,1,1)
        # plt.plot(history.history['loss'], label='loss')
        # plt.title('Loss history')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.xticks(range(0,20,2))
        # plt.legend(['training', 'validation'])

        # ax2 = plt.subplot(2,1,2)
        # plt.plot(history.history['accuracy'], label='accuracy')
        # plt.title('Accuracy history')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.xticks(range(0,20,2))

        # plt.tight_layout()
        # plt.show()
        # # plt.savefig('Training history.png')
    

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # normalize data
        X_clean = self._preprocessor(X_raw)

        # predict a label of 0 or 1
        return self.model.predict_classes(X_clean)

    def evaluate_architecture(self, X_raw, y_raw):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        # normalize data
        X_clean = self._preprocessor(X_raw)

        X = X_clean.to_numpy()
        y = y_raw.to_numpy()

        # k-fold cross-validation
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)

        # lists to save all false positive rates, true positive rates and aucs 
        fprs = []
        tprs = []
        aucs = []

        # perform k-fold cross-validation for AUC score
        for train, test in skf.split(X, y):
            # reset model for training
            self.reset()

            # train the model
            self.fit(X[train], y[train])

            # get prediction on test data
            y_pred = self.model.predict(X[test]).ravel()

            # get roc_curve
            fpr, tpr, thresholds = roc_curve(y[test], y_pred)

            # print(classification_report(y[test], self.predict(X[test])))

            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(auc(fpr, tpr))
        
        # avg_fpr = np.array(fprs).mean(axis=0)
        # avg_tpr = np.array(tprs).mean(axis=0)
        avg_auc = np.array(aucs).mean(axis=0)

        ############ plot auc-roc curve ##############
        # plt.clf()
        # plt.figure(1)
        # plt.plot([0, 1], [0, 1], 'k--')
        # for i in range(len(fprs)):
        #     plt.plot(fprs[i], tprs[i], label='Fold {} (area = {:.3f})'.format(i+1, aucs[i]))
        # plt.xlabel('False positive rate')
        # plt.ylabel('True positive rate')
        # plt.title('ROC curve for all cross-validation folds (avg. area = {:3f})'.format(avg_auc))
        # plt.legend(loc='best')
        # # plt.show()
        # plt.savefig('AUC-ROC.png')

        avg_auc = np.array(aucs).mean()
        print('Average AUC-ROC: {:.3f}'.format(avg_auc))

        return avg_auc

    def save_model(self):
        # with open("part2_claim_classifier.pickle", "wb") as target:
        #     pickle.dump(self, target)

        # use keras save model
        self.model.save('model.h5')


# This function might takes forever to run because it is training and evaluating over 400 models and pick the best one
# The function iterates through different chosen hyperparameters and save the model with the highest AUC-ROC score
def ClaimClassifierHyperParameterSearch(X_raw, y_raw):  # ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """
    # normalize data
    X = preprocessing.scale(X_raw)
    y = y_raw

    # lists of hyperparameters we want to search
    num_hidden_layers = [2, 3]
    num_hidden_neurons = [30, 60]
    dropout = [0.1, 0.2]
    lr = [0.001, 0.01]
    optimizer = ['Adam', 'RMSprop']
    weight_init = ['glorot_normal', 'he_normal']
    epochs = [10, 20]
    batch_size = [20, 40]

    # initialize best auc and params used for searching best model
    best_auc = -float('inf')
    best_params = None
    
    # loop through different hyperparameters to find the best model
    for nhl in num_hidden_layers:
        for nhn in num_hidden_neurons:
            for dr in dropout:
                for l in lr:
                    for opt in optimizer:
                        for wi in weight_init:
                            for ep in epochs:
                                for bs in batch_size:
                                    # create ClaimClassifier instance for evaluating hyperparameters
                                    model = ClaimClassifier(num_hidden_layers=nhl, num_hidden_neurons=nhn, dropout=dr, lr=l, weight_init=wi, optimizer=opt, num_epochs=ep, batch_size=bs)
                                    
                                    # evaluate model
                                    auc = model.evaluate_architecture(X, y)
                                    print('current auc: {}'.format(auc))

                                    # compare current model's auc to the best one so far
                                    # save model and parameters, and update the best auc
                                    if auc > best_auc:
                                        print('Current best auc: {}'.format(auc))
                                        model.save_model()
                                        best_auc = auc
                                        best_params = (nhl, nhn, dr, l, opt, wi, ep, bs)

    return  best_params

def load_model():
    """ Load saved model named 'model.h5'
    Note that you need to have the model.h5 somewhere in current directory
    return: instance of ClaimClassifier with model
    """
    return ClaimClassifier(load_keras_model('model.h5'))


if __name__ == '__main__':
    data = pd.read_csv('./part2_data.csv')
    
    x = data.drop(['claim_amount', 'made_claim'], axis=1)
    y = data[['made_claim']]
    
    # ClaimClassifierHyperParameterSearch(x,y)

    model = ClaimClassifier()
    model.evaluate_architecture(x,y)

