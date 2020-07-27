# CO553 - Intro to Machine Learning CW 1

This [README.md](https://github.com/paweenpit/CO553-Intro-to-ML-CW1/blob/master/README.md) explains how to run the code submitted for the CO553 Intro to Machine Learning coursework 1: Desicion Trees, from the group comprising of Niusha Alavi, Olle Nilsson, Paween Pitimanaaree and Alfred Tingey.

## Important Files

- [desicion_tree.py](https://github.com/paweenpit/CO553-Intro-to-ML-CW1/blob/master/decision_tree.py) contains the implementation of the algorithm for training desicion trees including the decision_tree_learning(dataset, depth) function as asked in the spec.

- [evaluation.py](https://github.com/paweenpit/CO553-Intro-to-ML-CW1/blob/master/evaluation.py) contains the implementation of crossvalidation and pruning algorithms. including the evaluate(test_db, trained_tree) function as asked in the spec.

- [visualize.py](https://github.com/paweenpit/CO553-Intro-to-ML-CW1/blob/master/visualize.py) contains the implementation of decision tree visulisation.

- [util.py](https://github.com/paweenpit/CO553-Intro-to-ML-CW1/blob/master/util.py) is for utilites such as functions for importing data.

- All other files (with .py extention) are setup to run the code to replicate the results in our report and to allow the marker to run our implemetation on other datasets.

## How to run the code?

After extracting the zip file, navigate to the extracted folder in the directory.

To run the code that we used to calculate the result that are in the report, run

```
python3 main.py 
```

This will first run the 10-fold cross-validation on both clean and noisy datasets and then run the 10-fold cross-validation with pruning on the clean and noisy datasets. Results will print in the terminal while it calculates and averages displayed at the end. Note that the dataset is shuffled in a random order each time so results may not match exacly to what's in the report.


## To run our crossvalidation code on a different dataset:

Move the file containing the dataset you wish to run to the extracted folder. Then run

```
python3 run_crossvalidation_on_unseen_data.py <yourdataset>
```

Be sure to replace `<yourdataset>` with your test dataset filename and extention(for example: data.txt). This will run the 10-fold crossvalidation on your new dataset and print results at the end.


## To run our crossvalidation code with pruning on a different dataset:

Move the file containg the dataset to the extracted folder. Then run

```
python3 run_pruning_on_unseen_data.py <yourdataset>
```

Be sure to replace `<yourdataset>` with your test dataset filename and extention(for example: data.txt). This will run 10-fold crossvalidation with pruning on the clean and noisy datasets. Results will print in screen while it calculates and averages displayed at the end.


## To run classification on our 'best' tree on dataset:

Move the file containg the dataset to the extracted folder. Then run

```
python3 run_classification.py <yourdataset>
```

Be sure to replace `<yourdataset>` with your test dataset filename and extention(for example: data.txt). This will first classify the samples in the dataset, with our 'best' trained tree using the evaluate(test_db, trained_tree) function, and display the results.
