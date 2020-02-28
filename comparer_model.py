import torch
import torch.nn.functional as F
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        print("11")
        super(EncoderRNN, self).__init__()
        print("12")
        self.hidden_size = hidden_size
        print("13")
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        print("14")
        self.gru = nn.GRU(hidden_size, hidden_size)
        print("15")

    def forward(self, _input, hidden):
        embedded = self.embedding(_input).view(1, _input.shape[0], -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, 1)

    def forward(self, _input, hidden, encoder_outputs):
        embedded = self.embedding(_input).view(1, 1, -1)
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

        output = self.out(output[0])
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
