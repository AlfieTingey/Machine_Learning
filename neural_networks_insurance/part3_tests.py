import pickle
import sklearn.metrics as met
import os
import pandas as pd
import numpy as np

import traceback
from pricing_library import MarketEvaluationTools

pd.options.mode.chained_assignment = (
    None
)  # to silence a particular panadas warning


class AImarket:
    def __init__(self, AI_price_matrix, X_market, claims_market, X_code_test):
        assert AI_price_matrix.shape[0] == len(X_market)
        market_matrix = np.zeros(
            shape=(len(X_market), AI_price_matrix.shape[1] + 1)
        )
        market_matrix[:, :-1] = AI_price_matrix
        self.price_matrix = market_matrix
        self.X = X_market
        self.claims = claims_market
        self.X_code_test = X_code_test

    def check_student_model_runs(self, std_model):
        try:
            prices = std_model.predict_premium(self.X_code_test)
        except Exception:
            print(
                "Could not generate prices using the 'predict_premium()' "
                "method"
            )
            s = traceback.format_exc(-1)
            print(s)
            return False
        try:
            # check is array:
            assert type(prices) == np.ndarray
        except Exception:
            print("Model does not output numpy array")
            s = traceback.format_exc(-1)
            print(s)
            return False
        try:
            # check length:
            assert len(prices) == len(self.X_code_test)
        except Exception:
            print("Model does not output array of correct length")
            s = traceback.format_exc(-1)
            print(s)
            return False
        try:
            # check is numerical array
            assert np.issubdtype(prices.dtype, np.number)
        except Exception:
            print("Output array elements are not numerical")
            s = traceback.format_exc(-1)
            print(s)
            return False
        return True

    def generate_student_model_prices(self, std_model):
        return std_model.predict_premium(self.X)

    def generate_market_summary(self, std_model, print_output=True):
        self.check_student_model_runs(std_model)
        prices = self.generate_student_model_prices(std_model)
        self.price_matrix[:, -1] = prices

        model_names = [
            "AI {}".format(i) for i in range(1, self.price_matrix.shape[1])
        ] + ["Your model"]
        mtools = MarketEvaluationTools(
            model_names=model_names,
            prices_df=pd.DataFrame(self.price_matrix, columns=model_names),
            claims=pd.Series(self.claims),
        )

        market_results = pd.DataFrame(index=model_names)
        market_results["profit"] = mtools.get_profit_per_firm()
        market_results["revenue"] = mtools.get_revenue_per_firm()
        market_results["total_loss"] = mtools.get_loss_per_firm()
        market_results["market_share"] = mtools.get_market_shares()
        market_results[
            "mean_price_offered"
        ] = mtools.get_mean_price_offered_per_firm()
        market_results["mean_price_won"] = mtools.get_mean_price_won_per_firm()
        market_results["mean_loss_in_market"] = np.array(
            [np.mean(self.claims)] * self.price_matrix.shape[1]
        )
        market_results[
            "mean_loss_incurred"
        ] = mtools.get_loss_per_contract_per_firm()
        market_results = market_results.sort_values(
            by="profit", ascending=False
        )
        market_results["rank"] = np.arange(1, len(model_names) + 1)
        cols = market_results.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        if print_output:
            print(market_results[cols])
        return market_results[cols]

    def check_student_rank_thershold(self, std_model, rank_threshold=5):
        market_results = self.generate_market_summary(std_model, False)
        market_rank = np.where(market_results.index == "Your model")[0][0] + 1
        print("YOUR RANK IN MARKET IS: {}".format(market_rank))
        print("THE REQUIRED RANK IS {} OR LOWER".format(rank_threshold))
        if market_results.loc["Your model", "market_share"] < 1e-5:
            print("YOUR MODEL DOES NOT CAPTURE ANY CONTRACTS!")
            return False
        return market_rank <= rank_threshold


def AUC_test(y_true, y_score, thresh=0.6):
    auc = met.roc_auc_score(y_true, y_score)
    print("YOUR MODEL AUC IS: {:.2f}".format(auc))
    print("TARGET AUC IS: {:.2f}".format(thresh))
    return auc >= thresh


# some necessary functions for the AI market
def mean_fitted_args(X, y):
    return np.mean(y)


def mean_decision_func(X, args):
    return np.ones(len(X)) * args


def pr_decision_base(X, premiums, args=None, profit_thresh=0):
    # this just adds profit_thresh % to all the revenues
    return premiums * (1 + profit_thresh)


def tests_part3(is_public, load_function=None, load_function_linear=None):
    test_results = {
        "Part 3 -- Linear model": {
            "score": 0,
            "name": "Part 3 -- Linear model",
            "possible": 3,
        },
        "Part 3 -- Nonlinear model": {
            "score": 0,
            "name": "Part 3 -- Nonlinear model",
            "possible": 3,
        },
    }
    code_runs_lin = False
    good_market_rank_lin = False
    good_auc_lin = False
    code_runs_non = False
    good_market_rank_non = False
    good_auc_non = False

    if is_public:
        lin_fname = "lin_AI_train.pickle"
        nonline_fname = "nonlin_AI_train.pickle"
        dat_fname = "part3_data.csv"
    else:
        lin_fname = "lin_AI_test.pickle"
        nonline_fname = "nonlin_AI_test.pickle"
        dat_fname = "part3_test_data.csv"

    try:
        with open(lin_fname, "rb") as source:
            lin_AI_mark = pickle.load(source)

        with open(nonline_fname, "rb") as source:
            nonlin_AI_mark = pickle.load(source)

        # from part3_pricing_model import PricingModel

        dat = pd.read_csv(dat_fname)
        X = dat.drop(columns=["claim_amount", "made_claim"])
        y = dat["made_claim"]

        # Loading the student models
        linear_exists = os.path.exists("part3_pricing_model_linear.pickle")
        nonlinear_exists = os.path.exists("part3_pricing_model.pickle")

        no_uploads_flag = True

        if linear_exists or (load_function_linear is not None):
            if linear_exists and (load_function_linear is None):
                with open("part3_pricing_model_linear.pickle", "rb") as source:
                    mymod = pickle.load(source)
            elif load_function_linear is not None:
                mymod = load_function_linear()

            no_uploads_flag = False

            print('SUMMARY OF "part3_pricing_model_linear.pickle"')
            code_runs_lin = lin_AI_mark.check_student_model_runs(
                mymod
            )  # binary output
            _ = lin_AI_mark.generate_market_summary(
                mymod
            )  # returns and prints dataframe
            good_market_rank_lin = lin_AI_mark.check_student_rank_thershold(
                mymod
            )  # binary output + print rank information
            good_auc_lin = AUC_test(
                y, mymod.predict_claim_probability(X))  # binary output

            print("=====================================")
            print("Current status on TRAINING data:")
            print("Code runs correctly:        {}".format(code_runs_lin))
            print("Market rank is acceptable:  {}".format(
                good_market_rank_lin))
            print("Train AUC passes threshold: {}".format(good_auc_lin))

        if nonlinear_exists or (load_function is not None):
            if nonlinear_exists and (load_function is None):
                with open("part3_pricing_model.pickle", "rb") as source:
                    mymod = pickle.load(source)
            elif load_function is not None:
                mymod = load_function()

            no_uploads_flag = False

            print('\nSUMMARY OF "part3_pricing_model.pickle":')

            code_runs_non = nonlin_AI_mark.check_student_model_runs(
                mymod
            )  # binary output
            _ = nonlin_AI_mark.generate_market_summary(
                mymod
            )  # returns and prints dataframe
            good_market_rank_non = nonlin_AI_mark.check_student_rank_thershold(
                mymod
            )  # binary output + print rank information
            good_auc_non = AUC_test(
                y, mymod.predict_claim_probability(X))  # binary output

            print("=====================================")
            print("Current status on TRAINING data:")
            print("Code runs correctly:        {}".format(code_runs_non))
            print("Market rank is acceptable:  {}".format(good_market_rank_non))
            print("Train AUC passes threshold: {}".format(good_auc_non))

        if no_uploads_flag:
            print(
                "NO MODEL FOUND. THIS COULD BE DUE TO:\n"
                "1. IF UPLOADING A PICKLE FILE, THE NAMEING CONVENTION IS WRONG\n"
                "2. IF USING AN ALTERNATIVE load_model FUNCTION, IT HAS NOT BEEN PROVIDED"
            )

    except Exception:
        print("PART 3: FAILED TO RUN TESTS.")
        s = traceback.format_exc(-1)
        print(s)

    n_correct_lin = sum([code_runs_lin, good_market_rank_lin, good_auc_lin])
    n_correct_non = sum([code_runs_non, good_market_rank_non, good_auc_non])

    test_results["Part 3 -- Linear model"]["score"] += n_correct_lin
    test_results["Part 3 -- Nonlinear model"]["score"] += n_correct_non

    return list(test_results.values())
