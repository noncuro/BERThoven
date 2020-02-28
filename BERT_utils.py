import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from tqdm import tqdm_notebook as tqdm
from transformers import (AdamW, AutoModel, AutoTokenizer, BertModel,
                          BertTokenizer, get_linear_schedule_with_warmup)
from utils import add_mask, pad, prepro_df

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


def get_new_bert_model():
    if not os.path.exists("bert_weights"):
        os.mkdir("bert_weights")
        bm = AutoModel.from_pretrained(
            "bert-base-multilingual-cased", force_download=True
        )
        bm.save_pretrained("./bert_weights/")
        torch.save(bm.state_dict(), "./bert_weights/bert-base-untrained.pth")
        print("Bert Model downloaded.")
    else:
        bm = AutoModel.from_pretrained("./bert_weights/")
        bm.load_state_dict(torch.load("./bert_weights/bert-base-untrained.pth"))
        print("Loaded pre-trained Bert weights.")
    assert is_model_new(bm)
    time.sleep(0.1)
    return bm


def is_model_new(bm: AutoModel):
    l = list(bm.parameters())
    return (
        l[6][13].item() == -0.11790694296360016
        and l[-5][10].item() == -0.015535828657448292
    )


class BERTHovenDataset(Dataset):
    def __init__(self, dataframe, test=False):
        self.samples = []
        self.test = test
        input1, input2 = get_tokenized(dataframe)
        x1, x1_mask = pad(input1)
        x2, x2_mask = pad(input2)
        for idx, _ in enumerate(tqdm(range(len(x1)), desc="Loading Data", leave=False)):
            sample = {
                "x1": x1[idx],
                "x1_mask": x1_mask[idx],
                "x2": x2[idx],
                "x2_mask": x2_mask[idx],
            }
            if not self.test:
                sample["score"] = dataframe.iloc[idx].scores
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
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
    """Dataset for image segmentation."""

    def __init__(self, dataframe, number_of_mask=1, test=False):
        super().__init__(dataframe, test)
        self.no_replace = [104, 102, 103, 0]  # MASK CLS SEP PAS
        self.number_of_mask = number_of_mask

    def __getitem__(self, item):
        sample = self.samples[item]
        x1 = self.add_mask(sample["x1"])
        x1_mask = sample["x1_mask"]
        x2 = self.add_mask(sample["x2"])
        x2_mask = sample["x2_mask"]

        if not self.test:
            score = sample["score"]
            return x1, x1_mask, x2, x2_mask, score
        return x1, x1_mask, x2, x2_mask

    def add_mask(self, x):
        for k in range(self.number_of_mask):
            index_mask = np.random.randint(0, len(x) - 1)
            while x[index_mask] in self.no_replace:
                index_mask = np.random.randint(0, len(x) - 1)
            x[index_mask] = 104

        return x


def get_tokenized(dataframe):
    input1 = (
        dataframe.apply(
            lambda a: "[CLS] " + a.src + " [SEP] " + a.mt + " [SEP]", axis=1
        )
        .apply(lambda a: tokenizer.tokenize(a))
        .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )

    input2 = (
        dataframe.apply(
            lambda a: "[CLS] " + a.mt + " [SEP] " + a.src + " [SEP]", axis=1
        )
        .apply(lambda a: tokenizer.tokenize(a))
        .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )
    return input1, input2


def get_tokenized_with_mask(dataframe):
    input1 = (
        dataframe.apply(
            lambda a: "[CLS] " + a.src + " [SEP] " + a.mt + " [SEP]", axis=1
        )
        .apply(lambda a: tokenizer.tokenize(a))
        .apply(lambda a: add_mask(a))
        .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )

    input2 = (
        dataframe.apply(
            lambda a: "[CLS] " + a.mt + " [SEP] " + a.src + " [SEP]", axis=1
        )
        .apply(lambda a: tokenizer.tokenize(a))
        .apply(lambda a: add_mask(a))
        .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )

    return input1, input2


def get_data_loader(dataframe, batch_size=32, test=False, preprocessor=None, fit=False):
    dataframe = prepro_df(dataframe, preprocessor, fit)
    ds = BERTHovenDataset(dataframe, test=test)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=(not test))


def get_data_loader_masked(
    dataframe, batch_size=32, test=False, preprocessor=None, fit=False
):
    dataframe = prepro_df(dataframe, preprocessor, fit)
    masked_df = MaskedDataset(dataframe)
    return torch.utils.data.DataLoader(
        masked_df, batch_size=batch_size, shuffle=(not test)
    )


def get_sentence_embeddings(dataframe, bert_model, device, test=False, batch_size=32):
    print("Tokenizing data...")
    input1, input2 = get_tokenized(dataframe)

    loader = torch.utils.data.DataLoader(list(zip(input1, input2)), batch_size=32)

    bert_model.eval()
    bert_model.to(device=device)

    l = []
    z = len(list(loader))

    with torch.no_grad():
        for i, (x1, x2) in enumerate(loader):
            x1 = torch.LongTensor(pad(x1)).to(device=device, dtype=torch.long)
            x2 = torch.LongTensor(pad(x2)).to(device=device, dtype=torch.long)
            o1 = bert_model(x1)[1]
            o2 = bert_model(x2)[1]

            out = [(o1[i], o2[i]) for i in range(len(o1))]
            l += out

    if not test:
        l = list(zip(l, dataframe.scores))
    return torch.utils.data.DataLoader(l, batch_size=batch_size, shuffle=(not test))
