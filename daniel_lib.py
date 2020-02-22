from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased")


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
        dataframe.apply(lambda a: "[CLS] " + a.src + " [SEP] " + a.mt + " [SEP]", axis=1)
            .apply(lambda a: tokenizer.tokenize(a))
            .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )

    input2 = (
        dataframe.apply(lambda a: "[CLS] " + a.mt + " [SEP] " + a.src + " [SEP]", axis=1)
            .apply(lambda a: tokenizer.tokenize(a))
            .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )
    return input1, input2


def get_tokenized_with_mask(dataframe):
    input1 = (
        dataframe.apply(lambda a: "[CLS] " + a.src + " [SEP] " + a.mt + " [SEP]", axis=1)
            .apply(lambda a: tokenizer.tokenize(a))
            .apply(lambda a: add_mask(a))
            .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )

    input2 = (
        dataframe.apply(lambda a: "[CLS] " + a.mt + " [SEP] " + a.src + " [SEP]", axis=1)
            .apply(lambda a: tokenizer.tokenize(a))
            .apply(lambda a: add_mask(a))
            .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )

    return input1, input2


def getDataLoader(dataframe, batch_size=32, test=False):
    input1, input2 = get_tokenized(dataframe)
    x1, x1_mask = pad(input1)
    x2, x2_mask = pad(input2)
    if test:
        l = list(zip(x1, x1_mask, x2, x2_mask))
    else:
        l = list(zip(x1, x1_mask, x2, x2_mask, dataframe.scores))

    return torch.utils.data.DataLoader(l, batch_size=batch_size, shuffle=(not test))


def getDataLoader_with_mask(dataframe, batch_size=32, test=False):
    input1, input2 = get_tokenized_with_mask(dataframe)
    x1, x1_mask = pad(input1)
    x2, x2_mask = pad(input2)
    if test:
        l = list(zip(x1, x1_mask, x2, x2_mask))
    else:
        l = list(zip(x1, x1_mask, x2, x2_mask, dataframe.scores))

    return torch.utils.data.DataLoader(l, batch_size=batch_size, shuffle=(not test))


def removeOutliers(dataframe, negLimit=-3, posLimit=2):
    dataframe.loc[dataframe.scores < negLimit, "scores"] = negLimit
    dataframe.loc[dataframe.scores > posLimit, "scores"] = posLimit
    return dataframe


from IPython.display import HTML, display


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


def get_sentence_embeddings(dataframe, bert_model, device, test=False, batch_size=32):
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

    def __init__(self,
                 sum_outputs=False,
                 concat_outputs=False,
                 cls=False,
                 dropout=True,
                 dropout_prob=0.5):
        super(BERThoven, self).__init__()
        if sum_outputs and concat_outputs:
            raise RuntimeError("You can't both sum and concatenate outputs.")

        self.bert_layers = AutoModel.from_pretrained("bert-base-multilingual-cased")
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
        out3 = self.lin_layer(out2)
        return out3.squeeze()


def check_accuracy(loader, model, device, max_sample_size=None):
    model = model.to(device=device)
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    abs_error = 0
    sqr_error = 0

    with torch.no_grad():
        scores_epoch = []
        truth_epoch = []

        for x1, x1_mask, x2, x2_mask, y in loader:
            truth_epoch += y.tolist()
            x1 = x1.to(device=device, dtype=torch.long)
            x1_mask = x1_mask.to(device=device, dtype=torch.long)
            x2 = x2.to(device=device, dtype=torch.long)
            x2_mask = x2_mask.to(device=device, dtype=torch.long)
            y = y.to(device=device, dtype=torch.float32)
            scores = model.forward((x1, x1_mask), (x2, x2_mask))
            scores_epoch += scores.tolist()

            abs_error += (scores - y).abs().sum()
            sqr_error += ((scores - y) ** 2).sum()
            num_samples += scores.size(0)
            if max_sample_size != None and num_samples >= num_samples:
                break
        mse = sqr_error / num_samples
        mae = abs_error / num_samples
        pr, _ = scipy.stats.pearsonr(scores_epoch, truth_epoch)

        print("Mean Absolute Error: %.3f, Mean Squared Error %.3f, Pearson: %.3f" % (mse, mae, pr))
    return mse, mae, pr


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
):
    # see F.smooth_l1_loss

    avg_loss = None
    momentum = 0.05

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print(f"Iterations per epoch:{len(dataloader)}")
        for t, (x1, x1_mask, x2, x2_mask, y) in enumerate(dataloader):
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
            if avg_loss is None:
                avg_loss = l
            else:
                avg_loss = l * momentum + avg_loss * (1 - momentum)

            if t % print_every == 0:
                print()
                print("Epoch: %d,\tIteration %d,\tloss = %.4f,\tavg_loss = %.4f" % (e, t, l, avg_loss), end="")
            print(".", end="")
        print()
        print("Avg loss %.3f" % (avg_loss))
        print("Checking accuracy on dev:")
        check_accuracy(val_loader, model, device=device)
        # print("Saving the model.")
        # torch.save(model.state_dict(), 'nlp_model.pt')
    return check_accuracy(val_loader, model, device=device)


def get_test_labels(loader, model, device):
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
    return all_scores