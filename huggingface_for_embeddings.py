# -*- coding: utf-8 -*-
"""HuggingFace for embeddings.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1L0_e7LWhX0OGlfAP8a7irUE3TVIq06pJ
"""

# !wget https://competitions.codalab.org/my/datasets/download/c748d2c0-d6be-4e36-9f12-ca0e88819c4d -O files.zip
# !unzip files.zip
# !pip install transformers

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from IPython.display import HTML, display
from torch import nn

from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased")

USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def import_file(prefix):
    with open(f"{prefix}.ende.src", "r", encoding="utf-8") as f:
        src = [line.strip() for line in f]
    with open(f"{prefix}.ende.mt", "r", encoding="utf-8") as f:
        mt = [line.strip() for line in f]
    scores = None
    if prefix != "test":
        with open(f"{prefix}.ende.scores", "r", encoding="utf-8") as f:
            scores = [float(line.strip()) for line in f]
    return pd.DataFrame({"src": src, "mt": mt, "scores": scores})


def pad(id_sequences, wiggle_room=0, max_length=None):
    if max_length is None:
        max_length = max([len(s) for s in id_sequences]) + wiggle_room
    padded_data = np.zeros((len(id_sequences), max_length))
    for i, sample in enumerate(id_sequences):
        padded_data[i, : len(sample)] = sample
    return padded_data


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


def getDataLoader(dataframe, batch_size=32, test=False):
    input1, input2 = get_tokenized(dataframe)
    if test:
        l = list(zip(pad(input1), pad(input2)))
    else:
        l = list(zip(pad(input1), pad(input2), dataframe.scores))

    return torch.utils.data.DataLoader(l, batch_size=batch_size, shuffle=(not test))


def removeOutliers(dataframe, negLimit=-3, posLimit=2):
    dataframe.loc[dataframe.scores < negLimit, "scores"] = negLimit
    dataframe.loc[dataframe.scores > posLimit, "scores"] = posLimit
    return dataframe


def progress(value, max=100):
    return HTML(
        """
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(
            value=value, max=max
        )
    )


def get_sentence_embeddings(dataframe, bert_model, test=False, batch_size=32):
    print("Tokenizing data...")
    input1, input2 = get_tokenized(dataframe)

    loader = torch.utils.data.DataLoader(list(zip(input1, input2)), batch_size=32)

    bert_model.eval()
    bert_model.to(device=device)

    l = []
    z = len(list(loader))

    progress_bar = display(progress(0, z), display_id=True)

    with torch.no_grad():
        for i, (x1, x2) in enumerate(loader):
            x1 = torch.LongTensor(pad(x1)).to(device=device, dtype=torch.long)
            x2 = torch.LongTensor(pad(x2)).to(device=device, dtype=torch.long)
            o1 = bert_model(x1)[1]
            o2 = bert_model(x2)[1]

            out = [(o1[i], o2[i]) for i in range(len(o1))]
            l += out

            progress_bar.update(progress(i, z))

    if not test:
        l = list(zip(l, dataframe.scores))
    return torch.utils.data.DataLoader(l, batch_size=batch_size, shuffle=(not test))


train_df = import_file("train")
dev_df = import_file("dev")
test_df = import_file("test")

dataLoader_train = getDataLoader(train_df)
dataLoader_dev = getDataLoader(dev_df)
dataLoader_test = getDataLoader(test_df, test=True)


train_df2 = pd.concat(
    [
        train_df,
        train_df[train_df.scores < -1],
        train_df[train_df.scores < -0.3],
        train_df[train_df.scores > 0.55],
        train_df[train_df.scores > 1],
        train_df[train_df.scores > 1.3],
    ]
)
# print(train_df2.shape)
# plt.subplot(2,1,1)
# train_df.scores.hist(bins=100)
# plt.subplot(2,1,2)
# train_df2.scores.hist(bins=100)
# print(train_df.scores.mean(), train_df2.scores.mean())
# print(train_df.scores.var(), train_df2.scores.var())
dataLoader_train2 = getDataLoader(train_df2)

train_df[train_df.scores < -5]

train_df[train_df.src == train_df.mt]
# train_df[train_df.scores>0.5]
train_df.scores.hist(bins=100)

train_df[train_df.scores < 0].shape


class BERThoven(nn.Module):
    def __init__(self, bert_model):
        super(BERThoven, self).__init__()
        self.bert_layers = bert_model
        bert_out_features = self.bert_layers.pooler.dense.out_features
        # self.lin_layer = nn.Linear(bert_out_features*2, 1)
        self.lin_layer = nn.Linear(bert_out_features, 1)

        self.droupout_layer = nn.Dropout(p=0.5)

    def forward(self, x1, x2):
        #        self.bert_layers.eval()
        #        with torch.no_grad():
        out1a = self.bert_layers(x1)[1]
        # out1b = self.bert_layers(x2)[1]

        # Using position 1 for the pooled head

        # out1x = torch.cat((out1a,out1b),1)
        out1x = out1a  # + out1b

        out2 = self.droupout_layer(out1x)
        out3 = self.lin_layer(out2)
        return out3.squeeze()


def check_accuracy(loader, model, max_sample_size=None):
    model = model.to(device=device)
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    abs_error = 0
    sqr_error = 0

    with torch.no_grad():
        scores_epoch = []
        truth_epoch = []

        for x1, x2, y in loader:
            truth_epoch += y.tolist()
            x1 = x1.to(device=device, dtype=torch.long)
            x2 = x2.to(device=device, dtype=torch.long)
            y = y.to(device=device, dtype=torch.float32)
            scores = model.forward(x1, x2)
            scores_epoch += scores.tolist()

            abs_error += (scores - y).abs().sum()
            sqr_error += ((scores - y) ** 2).sum()
            num_samples += scores.size(0)
            if max_sample_size != None and num_samples >= num_samples:
                break
        mse = sqr_error / num_samples
        mae = abs_error / num_samples
        pr, _ = scipy.stats.pearsonr(scores_epoch, truth_epoch)

        print(
            "Mean Absolute Error: %.3f, Mean Squared Error %.3f, Pearson: %.3f"
            % (mse, mae, pr)
        )


print_every = 75


def train_part(model, dataloader, optimizer, scheduler, epochs=1, max_grad_norm=1.0):

    avg_loss = 1
    momentum = 0.05

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print(f"Iterations per epoch:{len(dataloader)}")
        for t, (x1, x2, y) in enumerate(dataloader):
            model.train()  # put model to training mode
            x1 = x1.to(device=device, dtype=torch.long)
            x2 = x2.to(device=device, dtype=torch.long)
            y = y.to(device=device, dtype=torch.float32)

            scores = model(x1, x2)

            # loss = F.mse_loss(scores, y)
            loss = F.smooth_l1_loss(scores, y)
            # Zero out all of the gradients for the variables which the optimizer
            # will update.

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Effectively doubling the batch size
            # if t%2 ==0:
            #   torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            #   optimizer.step()
            #   optimizer.zero_grad()

            scheduler.step()
            l = loss.item()

            avg_loss = l * momentum + avg_loss * (1 - momentum)

            if t % print_every == 0:
                print()
                print(
                    "Epoch: %d, Iteration %d, loss = %.4f, avg_loss = %.4f"
                    % (e, t, l, avg_loss),
                    end="",
                )
            print(".", end="")
        print()
        print("Avg loss %.3f" % (avg_loss))
        print("Checking accuracy on dev:")
        check_accuracy(dataLoader_dev, model)
        # print("Saving the model.")
        # torch.save(model.state_dict(), 'nlp_model.pt')


def get_test_labels(loader, model):
    model = model.to(device=device)
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    abs_error = 0
    sqr_error = 0
    all_scores = []
    with torch.no_grad():
        for x1, x2 in loader:
            x1 = x1.to(device=device, dtype=torch.long)
            x2 = x2.to(device=device, dtype=torch.long)
            scores = model.forward(x1, x2)
            all_scores += [i.item() for i in scores]
    return all_scores


nlp_model = BERThoven(bert_model)
# nlp_model.load_state_dict(torch.load("nlp_model.pt"))

check_accuracy(dataLoader_dev, nlp_model)

epochs = 10
warmup_proportion = 0.1

steps_per_epoch = len(dataLoader_train2)
training_steps = steps_per_epoch * epochs
warmup_steps = int(training_steps * warmup_proportion)


optimizer = AdamW(nlp_model.parameters(), lr=1e-6, eps=1e-9, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
)

train_part(nlp_model, dataLoader_train2, optimizer, scheduler, epochs=epochs - 2)
train_part(nlp_model, dataLoader_train, optimizer, scheduler, epochs=2)

# check_accuracy(dataLoader_dev,nlp_model)

# def writeScores(scores):
#     fn = "predictions.txt"
#     print("")
#     with open(fn, 'w') as output_file:
#         for x in scores:
#             output_file.write(f"{x}\n")
# p.hist()
# dev_df.hist()
labels = get_test_labels(dataLoader_test, nlp_model)
p = pd.DataFrame(labels)  # .hist()
# p.head()
p.hist()

"""### Try using only embeddings"""


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.lin_layer1 = nn.Linear(768 * 2, 1000)
        self.relu1 = nn.LeakyReLU()
        self.lin_layer2 = nn.Linear(1000, 1000)
        self.relu2 = nn.LeakyReLU()
        self.lin_layer3 = nn.Linear(1000, 1)

    def forward(self, x1, x2):
        inp = torch.cat((x1, x2), 1)
        out1 = self.relu1(self.lin_layer1(inp))
        out2 = self.relu2(self.lin_layer2(out1))
        out3 = self.lin_layer3(out2)
        return out3.squeeze()


print_every = 500


def train_linear_model(
    model, dataloader, dataloader_val, optimizer, scheduler, epochs=1, max_grad_norm=1.0
):

    avg_loss = 1
    momentum = 0.01

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print(f"Iterations per epoch:{len(dataloader)}")
        for t, ((x1, x2), y) in enumerate(dataloader):
            model.train()  # put model to training mode
            y = y.to(device=device, dtype=torch.float32)

            scores = model(x1, x2)

            loss = F.mse_loss(scores, y)

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            scheduler.step()
            l = loss.item()

            avg_loss = l * momentum + avg_loss * (1 - momentum)

            if t % print_every == 0:
                print()
                print(
                    "Epoch: %d, Iteration %d, loss = %.4f, avg_loss = %.4f"
                    % (e, t, l, avg_loss),
                    end="",
                )
            print(".", end="")
        print()
        print("Avg loss %.3f" % (avg_loss))
        print("Checking accuracy on dev:")
        check_accuracy_linear(dataloader_val, model)
        # print("Saving the model.")
        # torch.save(model.state_dict(), 'nlp_model.pt')


def check_accuracy_linear(loader, model, max_sample_size=None):
    model = model.to(device=device)
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    abs_error = 0
    sqr_error = 0

    with torch.no_grad():
        for (x1, x2), y in loader:
            y = y.to(device=device, dtype=torch.float32)
            scores = model.forward(x1, x2)
            abs_error += (scores - y).abs().sum()
            sqr_error += ((scores - y) ** 2).sum()
            num_samples += scores.size(0)
            if max_sample_size != None and num_samples >= num_samples:
                break
        mse = sqr_error / num_samples
        mae = abs_error / num_samples
        print("Mean Absolute Error: %.3f, Mean Squared Error %.3f" % (mse, mae))


train_embeddings = get_sentence_embeddings(train_df, bert_model)
val_embeddings = get_sentence_embeddings(dev_df, bert_model)

epochs = 10
warmup_proportion = 0.1

steps_per_epoch = len(train_embeddings)
training_steps = steps_per_epoch * epochs
warmup_steps = int(training_steps * warmup_proportion)

linear_model = LinearModel()

optimizer = AdamW(linear_model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
)

train_linear_model(
    linear_model, train_embeddings, val_embeddings, optimizer, scheduler, epochs=epochs
)
