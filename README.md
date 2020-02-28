# BERThoven

In this repository, you will find the implementation of our different models that predicts the quality of translation between English and German. 
The dataset used to train these models constructed this way:
- src : Source document
- mt  : Machine translation
- scores : z-standardised score of three anotator's raw scores

The model outputs for each tupe (src, mt) a z-score


## Getting started
These instructions will get you a copy of the project up and running on your local machine

### Prerequisites
The requirements to run the project are the following
+ python 3.6
+ numpy
+ matplotlib
+ torch
+ tqdm
+ transformers
+ sklearn
+ scipy
+ pandas
+ requests

## Installation and usage


## Documentation

### plot-tree.py
##### `plot_tree(node)`: Plots a visualisation of the tree. 
- `node`: The decision tree to visualise

### kfold.py
##### `kfold(dataset, k_fold, run_pruning)`: Generate a decision tree using k-fold cross validation 
- `dataset`: 2D Numpy array. Dataset that will get split in the k-fold cross validation.
- `k_fold`: Integer. Desired number of folds in the dataset, where 1 fold will be used in cross validation, another will be kept out for testing, and the rest will be used for training
- `run_pruning`: Boolean. If True, decision trees will be pruned in the cross validation process

Returns: A tuple: `(models, training_sets)` where models is a list of the k models and training_sets is a list of the training sets left out in generating respective models

### decision_tree.py
##### `build_decision_tree(dataset,n_jobs)` Create a decision tree structure for a given data set.
- `dataset`: 2D Numpy array. Data set on which we want to train our model
- `n_jobs`: Integer, optional. How many threads to use, if multi-threading.

##### `build_tree_and_prune(training_set, validation_set)` Create a decision tree structure for a given data set and prune it afterwards.
- `training_set`: 2D Numpy array. Data set that will be used to train the decision tree
- `validation_set`: 2D Numpy array. Data set that will be used to in pruning.
- `n_jobs`: Integer, optional. How many threads to use, if multi-threading.

Returns: The pruned decision tree

### tree_structures.py
##### `Node.predict(dataset)`: Based on a trained decision tree, return the prediction of the given dataset.

- `Node`: the decision tree used to generate the predictions
- `dataset`: 2D Numpy array. The dataset on which predictions are desired

Returns: A list of the predictions, each corresponding with a row in `dataset`

##### `Node.predict_sample(sample)`: Based on a trained decision tree, return the prediction of a given sample.
- `sample`: 1D Numpy array. Sample that will get predicted by the model

Returns: the predicted label

### metrics.py

##### `evaluate(test_db, trained_tree)`: The accuracy of a tree, given a test dataset
- `test_db`: a 2D Numpy array to be used as a test set
- `trained_tree`: a Node to be used as a trained decision tree

Returns: The accuracy as a float between 0 and 1

##### `get_all_metrics(models, test_sets)`: Evaluate metrics following k-fold on the k models and k test sets
- `models`: list of Nodes.  Represents the decision trees to be evaluated (corresponding with the test_sets)
- `test_set`: list of 2D Numpy arrays. Each test set corresponds with a model, where that test set was omitted in k-fold when creating that model

Returns: A dictionary containing:
- `Confusion Matrix`: a 4x4 confusion matrix, calculated as the sum of the confusion matrices for each `test_set` and used for all other metric calculations
- `Average Accuracy`: The average accuracy across all `test_set`s
- `Precision`: a list with the precision for each class (i.e. room)
- `Average Precision`: The average precision across classes (i.e. rooms)
- `Recall`: a list with the recall for each class (i.e. room)
- `Average Recall`: The average recall across all classes (i.e. rooms)
- `F1`: a list with the F1 for each each class (i.e. room)
- `Average F1`: The average F1 across all classes (i.e. rooms)
- `Height`: a list with the maximum height for each model
- `Average Height`: the average height across the models 




