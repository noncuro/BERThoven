import torch
import torch.nn.functional as F
from torch import nn


class EncoderRNN(nn.Module):
    """
    Architecture for the encoder
    """

    def __init__(self, vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size  # Dimensions of the hidden state (memory vector)
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # Embedding layer
        self.gru = nn.LSTM(hidden_size, hidden_size)  # Recurrent layer

    def forward(self, _input, hidden):
        embedded = self.embedding(_input).view(1, _input.shape[0], -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    """
        Architecture for the decoder
        """

    def __init__(self, vocab_size, hidden_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size  # Dimensions of the hidden state (memory vector)
        self.vocab_size = vocab_size  # Number of words in the vocabulary
        self.dropout_p = dropout_p  # Dropout probability
        self.max_length = max_length  # Maximum length of a sentence

        self.embedding = nn.Embedding(
            self.vocab_size, self.hidden_size
        )  # Embedding layer
        self.attn = nn.Linear(
            self.hidden_size * 2, self.max_length
        )  # Attention generation layer
        self.attn_combine = nn.Linear(
            self.hidden_size * 2, self.hidden_size
        )  # Attention downscaling layer
        self.dropout = nn.Dropout(self.dropout_p)  # Dropout layer
        self.gru = nn.LSTM(self.hidden_size, self.hidden_size)  # Recurrent layer
        self.out = nn.Linear(self.hidden_size, 1)  # Final output generation layer

    def forward(self, _input, hidden, encoder_outputs):
        """
        Attention forward pass implemented according to Bahdanau et al on "NEURAL MACHINE TRANSLATION
        BY JOINTLY LEARNING TO ALIGN AND TRANSLATE"
        """
        embedded = self.embedding(_input).view(1, _input.shape[0], -1)
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

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
