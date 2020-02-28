import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import AdamW, AutoTokenizer
from utils import pad

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased")


def get_tokenized(dataframe):
    input1 = (
        dataframe.apply(lambda a: "[CLS] " + a.src, axis=1)
        .apply(lambda a: tokenizer.tokenize(a))
        .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )
    input2 = (
        dataframe.apply(lambda a: "[CLS] " + a.mt, axis=1)
        .apply(lambda a: tokenizer.tokenize(a))
        .apply(lambda a: tokenizer.convert_tokens_to_ids(a))
    )
    return input1, input2


class ComparerDataset(Dataset):
    """Dataset for image segmentation."""

    def __init__(self, dataframe, test=False):
        self.samples = []
        self.test = test
        input1, input2 = get_tokenized(dataframe)
        x1, x1_mask = pad(input1)
        x2, x2_mask = pad(input2)

        for idx in x1.shape[0]:
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
        base_item = (
            self.samples["x1"],
            self.samples["x1_mask"],
            self.samples["x2"],
            self.samples["x2_mask"],
        )
        if self.test:
            return base_item
        return (*base_item, self.samples["score"])


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
