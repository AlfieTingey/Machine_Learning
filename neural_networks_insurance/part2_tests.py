import os
import pickle
import pandas as pd
import numpy as np
import errno
import traceback

pd.options.mode.chained_assignment = None


def check_student_model_runs(std_model, X_sample):
    try:
        labels = std_model.predict(X_sample)
    except Exception:
        print('Could not generate predictions using the "predict()" method')
        s = traceback.format_exc(-1)
        print(s)
        return False
    try:
        # check is array:
        assert type(labels) == np.ndarray
    except Exception:
        print("Model does not output numpy array")
        s = traceback.format_exc(-1)
        print(s)
        return False
    try:
        # check length:
        assert len(labels) == len(X_sample)
    except Exception:
        print("Model does not output array of correct length")
        s = traceback.format_exc(-1)
        print(s)
        return False
    try:
        # check labels are 0 or 1
        all_labels = set(np.unique(labels))
        assert all_labels.issubset(set([0, 1]))
    except Exception:
        print("Output array elements are not 0 or 1")
        s = traceback.format_exc(-1)
        print(s)
        return False
    print("Your model runs and gives the correct output format")
    return True


def tests_part2(is_public, load_function=None):
    if is_public:
        fname = "part2_data.csv"
    else:
        fname = "part2_test_data.csv"

    test_results = {"Part 2": {"score": 0, "possible": 1, "name": "Part 2"}}

    # from part2_claim_classifier import ClaimClassifier
    try:
        dat = pd.read_csv(fname)
        X = dat.drop(columns=["claim_amount", "made_claim"])

        X_sample = X.sample(n=100, random_state=0)

        # Loading the student models
        file_exists = os.path.exists("part2_claim_classifier.pickle")

        if (not file_exists) and (load_function is None):
            print(
                "NO MODEL FOUND. THIS COULD BE DUE TO:\n"
                "1. IF UPLOADING A PICKLE FILE, THE NAMEING CONVENTION IS WRONG\n"
                "2. IF USING AN ALTERNATIVE load_model FUNCTION, IT HAS NOT BEEN PROVIDED"
            )
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                "part2_claim_classifier.pickle",
            )
        elif load_function is not None:
            mymod = load_function()
        else:
            with open("part2_claim_classifier.pickle", "rb") as source:
                mymod = pickle.load(source)

        test_result = check_student_model_runs(mymod, X_sample)

    except Exception:
        print("PART 2: FAILED TO RUN TESTS.")
        s = traceback.format_exc(-1)
        print(s)
        test_result = False

    test_results["Part 2"]["score"] += test_result

    return list(test_results.values())
