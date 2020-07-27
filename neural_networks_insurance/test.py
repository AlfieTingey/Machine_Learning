import pickle 

with open('part3_pricing_model.pickle', 'r') as target:
    classifier = pickle.load(target)
