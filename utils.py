import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm_notebook as tqdm
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

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


class BiLSTMDataset(Dataset):
    def __init__(self, dataframe, test=False):
        self.samples = []
        self.test = test
        src, mt = get_tokenized(dataframe)
        x1, _ = pad(src)
        x2, _ = pad(mt)
        for i, _ in enumerate(tqdm(range(len(x1)), desc="Loading Data", leave=False)):
            if self.test:
                self.samples.append((x1[i], x2[i]))
            else:
                self.samples.append((x1[i], x2[i], dataframe.iloc[i].scores))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]


def import_file(prefix, path="./"):
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
    train_df = import_file("train")
    dev_df = import_file("dev")
    ct = pd.concat([train_df, dev_df])

    return train_test_split(ct, shuffle=True, test_size=test_size)


def pad(id_sequences):
    max_length = max([len(s) for s in id_sequences])
    padded_data = np.zeros((len(id_sequences), max_length))
    mask = np.zeros_like(padded_data)
    for i, sample in enumerate(id_sequences):
        padded_data[i, : len(sample)] = sample
        mask[i, : len(sample)] = 1
    return padded_data, mask


def add_mask(sentence):
    index_mask = np.random.randint(0, len(sentence) - 1)
    while sentence[index_mask] == "[SEP]" or sentence[index_mask] == "[CLS]":
        index_mask = np.random.randint(0, len(sentence) - 1)

    sentence[index_mask] = "[MASK]"
    return sentence


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


def get_tokenized_one_way(dataframe):
    input1 = (
        dataframe.apply(lambda a: a.src, axis=1)
        .apply(lambda a: tokenizer.tokenize(a))
        .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )

    input2 = (
        dataframe.apply(lambda a: a.mt, axis=1)
        .apply(lambda a: tokenizer.tokenize(a))
        .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )
    return input1, input2


def prepro_df(dataframe, preprocessor, fit):
    if preprocessor:
        dataframe = dataframe.copy()
        scores = dataframe.scores.to_numpy().reshape(-1, 1)
        if fit:
            scores = preprocessor.fit_transform(scores)
        else:
            scores = preprocessor.transform(scores)
        dataframe.scores = scores
    return dataframe


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


def get_data_loader_bilstm(
    dataframe, batch_size=32, test=False, preprocessor=None, fit=False
):
    dataframe = prepro_df(dataframe, preprocessor, fit)
    ds = BiLSTMDataset(dataframe, test=test)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=(not test))


def remove_outliers(dataframe, negLimit=-3, posLimit=2):
    dataframe.loc[dataframe.scores < negLimit, "scores"] = negLimit
    dataframe.loc[dataframe.scores > posLimit, "scores"] = posLimit
    return dataframe


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


def augment_dataset(original, *score_lambdas):
    to_concat = [original]
    for i in score_lambdas:
        to_concat += [original[i(original.scores)]]
    return pd.concat(to_concat)


class BERThoven(nn.Module):
    """
        sum_outputs: run BERT with text in both directions, then sum the outputs for the fc layer
        concat_outputs: run BERT with text in both directions, then concatenate the outputs for the fc layer
        cls: use the [CLS] output (instead of the pooled output)
    """

    def __init__(
        self,
        sum_outputs=False,
        concat_outputs=False,
        cls=False,
        dropout=True,
        dropout_prob=0.5,
    ):
        super(BERThoven, self).__init__()
        if sum_outputs and concat_outputs:
            raise RuntimeError("You can't both sum and concatenate outputs.")

        self.bert_layers = get_new_bert_model()
        bert_out_features = self.bert_layers.pooler.dense.out_features
        if concat_outputs:
            self.lin_layer = nn.Linear(bert_out_features * 2, 1)
        else:
            self.lin_layer = nn.Linear(bert_out_features, 1)

        self.droupout_layer = nn.Dropout(p=dropout_prob)

        self.droupout = dropout
        self.both_ways = sum_outputs or concat_outputs
        self.sum_outputs = sum_outputs
        self.concat_outputs = concat_outputs
        self.get_bert_output = lambda x: x[0][:, 0, :] if cls else x[1]

    def forward(self, x1, x2):
        # The 1 index is for the pooled head
        out1a = self.bert_layers(x1[0], attention_mask=x1[1])
        out1a = self.get_bert_output(out1a)
        if not self.both_ways:
            out1x = out1a
        else:
            out1b = self.bert_layers(x2[0], attention_mask=x2[1])
            out1b = self.get_bert_output(out1b)
            if self.concat_outputs:
                out1x = torch.cat((out1a, out1b), 1)
            else:
                out1x = out1a + out1b

        out2 = self.droupout_layer(out1x) if self.droupout else out1x
        out2 = self.lin_layer(out2)
        out2 = torch.sigmoid(out2)
        return out2.squeeze()


def check_accuracy(loader, model, device, max_sample_size=None, preprocessor=None):
    model = model.to(device=device)
    num_samples = 0
    model.eval()  # set model to evaluation mode
    abs_error = 0
    sqr_error = 0

    with torch.no_grad():
        scores_epoch = []
        truth_epoch = []

        for x1, x1_mask, x2, x2_mask, y in tqdm(
            loader, "Checking accuracy", leave=False
        ):
            truth_epoch += y.tolist()
            x1 = x1.to(device=device, dtype=torch.long)
            x1_mask = x1_mask.to(device=device, dtype=torch.long)
            x2 = x2.to(device=device, dtype=torch.long)
            x2_mask = x2_mask.to(device=device, dtype=torch.long)
            scores = model.forward((x1, x1_mask), (x2, x2_mask))

            scores = scores.cpu().numpy().reshape(-1, 1)
            y = y.cpu().numpy().reshape(-1, 1)

            if preprocessor is not None:
                scores = preprocessor.inverse_transform(scores)
                y = preprocessor.inverse_transform(y)
            scores_epoch += scores.reshape(-1).tolist()

            abs_error += np.abs(scores - y).sum().item()
            sqr_error += ((scores - y) ** 2).sum().item()
            num_samples += scores.shape[0]
            if max_sample_size != None and num_samples >= num_samples:
                break
        rmse = (sqr_error / num_samples) ** 0.5
        mae = abs_error / num_samples
        pr, _ = scipy.stats.pearsonr(scores_epoch, truth_epoch)

        print(
            "Mean Absolute Error: %.3f, Root Mean Squared Error %.3f, Pearson: %.3f"
            % (rmse, mae, pr)
        )
    return rmse, mae, pr


def smoothing(l, w_size=3):
    l2 = []
    for i in range(0, len(l) - 2):
        l2.append(np.mean(l[i : i + w_size]))
        x = np.linspace(0, 1, len(l2))
    return x, l2


def train_part(
    model,
    dataloader,
    optimizer,
    scheduler,
    val_loader,
    device,
    epochs=1,
    max_grad_norm=1.0,
    print_every=75,
    loss_function=F.mse_loss,
    return_metrics=True,
    val_every=None,
    return_losses=False,
    preprocessor: QuantileTransformer = None,
):
    # see F.smooth_l1_loss

    avg_loss = None
    avg_val_loss = None
    momentum = 0.05

    t_losses = []
    v_losses = []

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print(f"Iterations per epoch:{len(dataloader)}")
        time.sleep(0.1)
        for t, (x1, x1_mask, x2, x2_mask, y) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            model.train()  # put model to training mode
            x1 = x1.to(device=device, dtype=torch.long)
            x1_mask = x1_mask.to(device=device, dtype=torch.long)
            x2 = x2.to(device=device, dtype=torch.long)
            x2_mask = x2_mask.to(device=device, dtype=torch.long)
            y = y.to(device=device, dtype=torch.float32)

            scores = model.forward((x1, x1_mask), (x2, x2_mask))

            loss = loss_function(scores, y)
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
            t_losses.append(l)
            if avg_loss is None:
                avg_loss = l
            else:
                avg_loss = l * momentum + avg_loss * (1 - momentum)

            if val_every is not None and t % val_every == 0:
                with torch.no_grad():
                    (x1, x1_mask, x2, x2_mask, y) = next(iter(val_loader))
                    x1 = x1.to(device=device, dtype=torch.long)
                    x1_mask = x1_mask.to(device=device, dtype=torch.long)
                    x2 = x2.to(device=device, dtype=torch.long)
                    x2_mask = x2_mask.to(device=device, dtype=torch.long)
                    y = y.to(device=device, dtype=torch.float32)
                    scores = model.forward((x1, x1_mask), (x2, x2_mask))
                    l_val = loss_function(scores, y).item()
                    v_losses.append(l_val)
                    if avg_val_loss is None:
                        avg_val_loss = l_val
                    else:
                        avg_val_loss = l_val * momentum + avg_val_loss * (1 - momentum)

            if t % print_every == 0:
                print()
                if avg_val_loss is not None:
                    print(
                        "Epoch: %d,\tIteration %d,\tMoving avg loss = %.4f\tval loss = %.4f"
                        % (e, t, avg_loss, avg_val_loss),
                        end="\t",
                    )
                else:
                    print(
                        "Epoch: %d,\tIteration %d,\tMoving avg loss = %.4f"
                        % (e, t, avg_loss),
                        end="\t",
                    )
            # print(".", end="")
        print()
        print("Checking accuracy on dev:")
        check_accuracy(val_loader, model, device=device, preprocessor=preprocessor)
        # print("Saving the model.")
        # torch.save(model.state_dict(), 'nlp_model.pt')
    if return_metrics:
        return check_accuracy(
            val_loader, model, device=device, preprocessor=preprocessor
        )
    if return_losses:
        px, py = smoothing(t_losses, 30)
        plt.plot(epochs * px, py, label="Training loss")
        px, py = smoothing(v_losses, 20)
        plt.plot(epochs * px, py, label="Validation loss")
        plt.legend()
        plt.show()
        return t_losses, v_losses


def get_test_labels(loader, model, device, preprocessor=None):
    model = model.to(device=device)
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    abs_error = 0
    sqr_error = 0
    all_scores = []
    with torch.no_grad():
        for x1, x1_mask, x2, x2_mask in loader:
            x1 = x1.to(device=device, dtype=torch.long)
            x1_mask = x1_mask.to(device=device, dtype=torch.long)
            x2 = x2.to(device=device, dtype=torch.long)
            x2_mask = x2_mask.to(device=device, dtype=torch.long)
            scores = model.forward((x1, x1_mask), (x2, x2_mask))
            all_scores += [i.item() for i in scores]
    if preprocessor is not None:
        all_scores = np.array(all_scores).reshape(-1, 1)
        all_scores = preprocessor.inverse_transform(all_scores).reshape(-1)
    return all_scores


def writeScores(scores):
    fn = "predictions.txt"
    print("")
    with open(fn, "w") as output_file:
        for idx, x in enumerate(scores):
            output_file.write(f"{x}\n")
