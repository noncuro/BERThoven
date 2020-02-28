import os

import numpy as np
import torch
from torch.utils.data import Dataset

from tqdm import tqdm_notebook as tqdm
from transformers import AutoModel, AutoTokenizer
from utils import add_mask, pad, prepro_df

# Tokenizer from pretrained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


def get_new_bert_model():
    """
    Returns a new BERT model with guaranteed freshly loaded weights.
    Avoids retraining models on top of others if you're not careful
    """

    # If the model isn't downloaded, download it and save it to disk
    if not os.path.exists("bert_weights"):
        os.mkdir("bert_weights")
        bm = AutoModel.from_pretrained(
            "bert-base-multilingual-cased", force_download=True
        )
        bm.save_pretrained("./bert_weights/")
        torch.save(bm.state_dict(), "./bert_weights/bert-base-untrained.pth")
        print("Bert Model downloaded.")
    # Otherwise, simply load it from disk
    else:
        bm = AutoModel.from_pretrained("./bert_weights/")
        bm.load_state_dict(torch.load("./bert_weights/bert-base-untrained.pth"))
        print("Loaded pre-trained Bert weights.")

    # In experimentation, we found that sometimes the BERT model would be reused between experiments.
    # We add this assertion just to be sure that doesn't happen.
    assert is_model_new(bm)
    return bm


def is_model_new(bm: AutoModel):
    """Verifies if the BERT model has fresh weights, i.e. hasn't been trained yet.
    bm: AutoModel object representing the BERT model
    """
    l = list(bm.parameters())
    # Check two arbitrary weights for equality
    # We simply check two weights for computational efficiency
    return (
            l[6][13].item() == -0.11790694296360016
            and l[-5][10].item() == -0.015535828657448292
    )


class BERTHovenDataset(Dataset):
    """Class responsible of handling pre-processing of data handed to BERT, for use within a dataloader
    """

    def __init__(self, dataframe, test=False):
        """
        dataframe: pandas.DataFrame object containing the dataset
        test: boolean values describing whether or not this is the test set
        """
        self.samples = []  # Will store the final data
        self.test = test  # Whether or not the dataset is a test set

        # Convert the strings to token indices
        input1, input2 = get_tokenized(dataframe)

        # Pad data according to the longest sentence
        x1, x1_mask = pad(input1)
        x2, x2_mask = pad(input2)

        # Build the tuple structure to be used in training and testing
        for idx, _ in enumerate(tqdm(range(len(x1)), desc="Loading Data", leave=False)):
            sample = {
                "x1": x1[idx],
                "x1_mask": x1_mask[idx],
                "x2": x2[idx],
                "x2_mask": x2_mask[idx],
            }

            # The score feature is only available for test and dev sets
            if not self.test:
                sample["score"] = dataframe.iloc[idx].scores
            self.samples.append(sample)

    def __len__(self):
        """Returns the size of the dataset"""
        return len(self.samples)

    def __getitem__(self, item):
        """Returns a tuple to be used in an iteration of training or testing
        item: index of the row to select the tuple from
        """
        sample = self.samples[item]
        x1 = sample["x1"]
        x1_mask = sample["x1_mask"]
        x2 = sample["x2"]
        x2_mask = sample["x2_mask"]
        if not self.test:
            score = sample["score"]
            return x1, x1_mask, x2, x2_mask, score
        return x1, x1_mask, x2, x2_mask


class MaskedDataset(BERTHovenDataset):
    """
    Class responsible of handling pre-processing of data handed to BERT. This modification allows for words in the
    sentence to randomly be replaced by [MASK] tokens.
    This can be seen as a way to introduce noise into the system, to reduce the likelihood of overfitting.
    """

    def __init__(self, dataframe, number_of_masks=1, test=False):
        """
        dataframe: pandas.DataFrame object containing the dataset
        number_of_masks: Number of words to substitute for masks in the sentence
        test: boolean values describing whether or not this is the test set
        """
        super().__init__(dataframe, test)

        # Special tokens shouldn't be replaced by mask
        self.no_replace = [104, 102, 103, 0]  # MASK CLS SEP PAS
        self.number_of_mask = number_of_masks  # Saves the number of masks

    def __getitem__(self, item):
        """Returns a tuple with masked sentences to be used in an iteration of
        training or testing
        item: index of the row to select the tuple from
        """

        # Select appropriate row according to `item`
        sample = self.samples[item]

        # Adds masks to the original sentence
        x1 = self.add_mask(sample["x1"])
        x1_mask = sample["x1_mask"]
        x2 = self.add_mask(sample["x2"])
        x2_mask = sample["x2_mask"]

        # Return tuple
        if not self.test:
            score = sample["score"]
            return x1, x1_mask, x2, x2_mask, score
        return x1, x1_mask, x2, x2_mask

    def add_mask(self, x):
        """Randomly substitutes tokens for masks according to a predefinied
        number of masks
        x: List of indices representing the original sentence
        """
        for k in range(self.number_of_mask):
            index_mask = np.random.randint(0, len(x) - 1)
            while x[index_mask] in self.no_replace:
                index_mask = np.random.randint(0, len(x) - 1)
            x[index_mask] = 104

        return x


def get_tokenized(dataframe):
    """Performs tokenization and index substitution on a dataframe
    This function also concatenates the sentences in both ways, appending a
    [SEP] token after each one.

    dataframe: pandas.DataFrame object containing the dataset
    """
    input1 = (
        dataframe.apply(
            # Source language followed by translation
            lambda a: "[CLS] " + a.src + " [SEP] " + a.mt + " [SEP]",
            axis=1,
        )
            # First we tokenize
            .apply(lambda a: tokenizer.tokenize(a))
            # Then we substitute indices
            .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )

    input2 = (
        dataframe.apply(
            # Translation followed by source language
            lambda a: "[CLS] " + a.mt + " [SEP] " + a.src + " [SEP]",
            axis=1,
        )
            .apply(lambda a: tokenizer.tokenize(a))  # Tokenize
            .apply(lambda a: tokenizer.convert_tokens_to_ids(a))  # Substitute indices
    )
    return input1, input2


def get_data_loader(dataframe, batch_size=32, test=False, preprocessor=None, fit=False):
    """Returns a Torch DataLoader object containing the dataset
    dataframe: pandas.DataFrame object containing the dataset
    batch_size: Size of the batches
    test: boolean value describing whether or not this is the test set
    preprocessor: A custom function to perform pre-processing on the data
    fit: Boolean value describing whether or not to fit the preprocessor to the data
    """
    # Start by pre-processing the data
    # TODO: Move outside the function
    dataframe = prepro_df(dataframe, preprocessor, fit)

    # Create an instance of the dataset wrapper
    ds = BERTHovenDataset(dataframe, test=test)

    # Return a Torch compatible DataLoader
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=(not test))


def get_data_loader_masked(
        dataframe, batch_size=32, test=False, preprocessor=None, fit=False
):
    """Returns a Torch DataLoader object containing the dataset
    This DataLoader applies random masking to its data
    dataframe: pandas.DataFrame object containing the dataset
    batch_size: Size of the batches
    test: boolean value describing whether or not this is the test set
    preprocessor: A custom function to perform pre-processing on the data
    fit: Boolean value describing whether or not to fit the preprocessor to the data
    """
    # Start by pre-processing the data
    dataframe = prepro_df(dataframe, preprocessor, fit)

    # Create an instance of the masking capable dataset wrapper
    masked_df = MaskedDataset(dataframe)

    # Return a Torch compatible DataLoader
    return torch.utils.data.DataLoader(
        masked_df, batch_size=batch_size, shuffle=(not test)
    )


def get_sentence_embeddings(dataframe, bert_model, device, test=False, batch_size=32):
    """Returns the embeddings of a sentence as the pooled outputs from a pretrained BERT model.
    dataframe: pandas.DataFrame object containing the dataset
    bert_model BERT model used to generate the embeddings
    device: The device to use. Either "cuda" or "cpu"
    test: boolean value describing whether or not this is the test set
    batch_size: Size of the batches
    """
    print("Tokenizing data...")
    # Start by tokenizeng the sentences
    input1, input2 = get_tokenized(dataframe)

    # Create a custom DataLoader to serve the sentence pair in question
    loader = torch.utils.data.DataLoader(list(zip(input1, input2)), batch_size=32)

    # Set model to evaluation mode
    bert_model.eval()

    # Set the model to the appropriate device
    bert_model.to(device=device)

    embeddings = []  # Will hold the embeddings

    # Deactivate gradients for evaluation
    with torch.no_grad():
        # Iterate over the temporary DataLoader to retrieve sentences
        for i, (x1, x2) in enumerate(loader):
            # Generate embeddings
            x1 = torch.LongTensor(pad(x1)).to(device=device, dtype=torch.long)
            x2 = torch.LongTensor(pad(x2)).to(device=device, dtype=torch.long)
            o1 = bert_model(x1)[1]
            o2 = bert_model(x2)[1]

            out = [(o1[i], o2[i]) for i in range(len(o1))]
            embeddings += out  # Store embeddings

    if not test:
        embeddings = list(zip(embeddings, dataframe.scores))
    # Return a new DataLoader containing the embeddings
    return torch.utils.data.DataLoader(
        embeddings, batch_size=batch_size, shuffle=(not test)
    )
