from sklearn.calibration import CalibratedClassifierCV

import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop



data = pd.read_csv('part3_data.csv', nrows =None)
y = data['made_claim']
y_claim = data['claim_amount']
unique, counts = np.unique(y, return_counts=True)

X = data.iloc[:,:-2]



def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


def main():

    model = PricingModel()
    model.training = True
    model.fit(X, y, y_claim)
    model.training = False
    prob = model.predict_claim_probability(X.iloc[:100,:])
    price = model.predict_premium(X)
    model.save_model()



# class for part 3.
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.drop = ['id_policy', 'town_mean_altitude', 'town_surface_area', 'commune_code',\
                'canton_code', 'city_district_code', 'regional_department_code', 'vh_model', 'pol_insee_code'
                ]

        self.numerical = ['pol_bonus', 'pol_duration', 'pol_sit_duration', 'drv_age1', 'drv_age2',\
                      'drv_age_lic1', 'drv_age_lic2', 'vh_age', 'vh_cyl', 'vh_din',\
                       'vh_sale_begin', 'vh_sale_end', 'vh_speed', 'vh_value', 'vh_weight',\
                       'town_mean_altitude', 'town_surface_area', 'population']

        self.categorical = ['pol_coverage', 'pol_pay_freq', 'pol_payd', 'pol_usage',\
                        'pol_insee_code', 'drv_drv2', 'drv_sex1', 'drv_sex2', 'vh_fuel',\
                        'vh_make', 'vh_model', 'vh_type', 'commune_code', 'canton_code',\
                         'city_district_code', 'regional_department_code']

        self.binary = ['pol_payd', 'pol_usage', 'drv_drv2', 'drv_sex1', 'drv_sex2','vh_type']


        self.outliers = ['drv_age_lic2']#columns with big outliers

        self.missing_values = ['drv_age2', 'drv_sex2', 'drv_age_lic2']#can have missing values

        self.class_weight = {0: 1.,
                        1: 9.}
        self.training = False

        self.y_mean = None
        self.calibrate = calibrate_probabilities
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.base_classifier =  None


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
    def _preprocessor(self, X_raw): # X_raw is a pandas object
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE
        if self.training:
            numeric_features = [x for x in self.numerical if x not in self.drop]
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])

            multi_categorical_features = [x for x in self.categorical if x not in\
                                                self.drop and x not in self.binary]
            multi_categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            binary_categorical_features = [x for x in self.binary if x not in self.drop]
            binary_categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ord', OrdinalEncoder())])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('one', multi_categorical_transformer, multi_categorical_features),
                    ('bin', binary_categorical_transformer, binary_categorical_features),
                    ])
            self.preprocessor=preprocessor.fit(X_raw)

        # X = preprocessor.fit_transform(X_raw)
        # feature_selec = VarianceThreshold(threshold=(.8 * (1 - .8)))
        return self.preprocessor.transform(X_raw)


    def get_model(self):
        inputs = Input(shape=(self.input_shape,))
        output_1 = Dense(64, activation='relu')(inputs)
        output_2 = Dense(32, activation='relu')(output_1)
        # output_3 = Dense(32, activation='relu')(output_2)

        predictions = Dense(1, activation='sigmoid')(output_2)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return(model)


    def fit(self, X_raw, y_raw, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        self.input_shape = np.shape(X_clean)[1]
        self.base_classifier= KerasClassifier(build_fn=self.get_model)

        # X_train, X_test, y_train, y_test =\
        # 	 	train_test_split(X_clean, y_raw, test_size=0.2)

        self.base_classifier.fit(X_clean, y_raw, batch_size=64,\
		epochs=10, validation_split=0.2, verbose = 2, class_weight=self.class_weight)



        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        # if self.calibrate:
        #     self.base_classifier = fit_and_calibrate_classifier(
        #         self.base_classifier, X_clean, y_raw)
        # else:
        #     self.base_classifier = self.base_classifier.fit(X_clean, y_raw, batch_size=64,\
        #     		epochs=10, validation_split=0.2, verbose = 1, class_weight=self.class_weight)
        return self.base_classifier


    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        return self.base_classifier.predict_proba(X_clean)[:,1]


    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

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
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor
        # X_clean = self._preprocessor(X_raw)
        # return X_clean

        return self.predict_claim_probability(X_raw) * self.y_mean*0.2

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)


if __name__ == '__main__':

    pass
    main()
