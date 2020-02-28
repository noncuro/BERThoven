import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def import_file(prefix, path="./"):
    """Helper function to generate a DataFrame from file
    prefix: String containing the desired type of dataset to load (one from `train`, `dev`, `test`)
    path: Path to file
    """
    with open(os.path.join(path, f"{prefix}.ende.src"), "r", encoding="utf-8") as f:
        src = [line.strip() for line in f]
    with open(os.path.join(path, f"{prefix}.ende.mt"), "r", encoding="utf-8") as f:
        mt = [line.strip() for line in f]
    scores = None
    if prefix != "test":
        with open(
            os.path.join(path, f"{prefix}.ende.scores"), "r", encoding="utf-8"
        ) as f:
            scores = [float(line.strip()) for line in f]
    return pd.DataFrame({"src": src, "mt": mt, "scores": scores})


def import_train_dev(test_size=1 / 8):
    """Imports both training and dev set, recombine them and generate another
    train/dev split
    test_size: Dev to train ratio
    """
    train_df = import_file("train")
    dev_df = import_file("dev")
    ct = pd.concat([train_df, dev_df])

    return train_test_split(ct, shuffle=True, test_size=test_size)


def pad(id_sequences):
    """HElper function to pad all sentences according to the longest one.
    id_sequences: Sequence of token indices representing sentences
    """
    max_length = max([len(s) for s in id_sequences])
    padded_data = np.zeros((len(id_sequences), max_length))
    mask = np.zeros_like(padded_data)
    for i, sample in enumerate(id_sequences):
        padded_data[i, : len(sample)] = sample
        mask[i, : len(sample)] = 1
    return padded_data, mask


def add_mask(sentence):
    """Helper function to randomly add a mask to a sentence
    sentence: Original sentence
    """
    index_mask = np.random.randint(0, len(sentence) - 1)
    while sentence[index_mask] == "[SEP]" or sentence[index_mask] == "[CLS]":
        index_mask = np.random.randint(0, len(sentence) - 1)

    sentence[index_mask] = "[MASK]"
    return sentence


def prepro_df(dataframe, preprocessor, fit):
    """Applies a preprocessing function to a DataFrame if one is provided
    dataframe: pandas.DataFrame object containing the dataset
    preprocessor: A function to perform pre-processing on the data
    fit: Boolean value describing whether or not to fit the preprocessor to the data
    """
    # If a preprocessor is supplied, then apply it
    if preprocessor:
        dataframe = dataframe.copy()
        scores = dataframe.scores.to_numpy().reshape(-1, 1)
        if fit:
            scores = preprocessor.fit_transform(scores)
        else:
            scores = preprocessor.transform(scores)
        dataframe.scores = scores
    # Otherwise return the original DataFrame
    return dataframe


def remove_outliers(dataframe, negLimit=-3, posLimit=2):
    """Helper function to remove outliers from the dataset
    This function removes datapoints that are outside of [negLimit, posLimit]
    dataframe: pandas.DataFrame object containing the dataset
    negLimit: Smallest accepted value (inclusive)
    posLimit: Largest accepted value (inclusive)
    """
    dataframe.loc[dataframe.scores < negLimit, "scores"] = negLimit
    dataframe.loc[dataframe.scores > posLimit, "scores"] = posLimit
    return dataframe


def augment_dataset(original, *score_lambdas):
    """Upsamples less represented bins of a histogram of the dataset
    original: pandas.DataFrame object containing the original dataset
    score_lambdas:
    """
    to_concat = [original]
    for i in score_lambdas:
        to_concat += [original[i(original.scores)]]
    return pd.concat(to_concat)


def smoothing(l, w_size=3):
    """Applies window smoothing
    l: List of values on which to apply smoothing
    w_size: Size of the window when performing smoothing
    """
    l2 = []
    for i in range(0, len(l) - 2):
        l2.append(np.mean(l[i : i + w_size]))
        x = np.linspace(0, 1, len(l2))
    return x, l2
