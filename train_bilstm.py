import time

import torch
import torch.nn as nn
from torch import optim

import comparer_model


class TrainerBiLSTM:
    def __init__(self, device):
        self.device = device

    def train_once(
        self,
        src_tensor,
        mt_tensor,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        loss_function,
        max_length,
    ):
        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        src_length = src_tensor.size(0)
        mt_length = mt_tensor.size(0)

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=self.device
        )

        loss = 0

        for i in range(src_length):
            encoder_output, encoder_hidden = encoder(src_tensor[i], encoder_hidden)
            encoder_outputs[i] = encoder_output[0, 0]

        decoder_hidden = encoder_hidden
        for i in range(mt_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                mt_tensor[i], decoder_hidden, encoder_outputs
            )
            loss += loss_function(decoder_output, mt_tensor[i])

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / mt_length

    def trainIters(
        self,
        encoder,
        decoder,
        epochs,
        print_every=1000,
        learning_rate=0.01,
    ):
        print_loss_total = 0  # Reset every print_every

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(epochs)]
        loss_function = nn.MSELoss()

        for iter in range(1, epochs + 1):
            training_pair = training_pairs[iter - 1]
            src_tensor = training_pair[0]
            mt_tensor = training_pair[1]

            loss = self.train(
                src_tensor,
                mt_tensor,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                loss_function,
            )
            print_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print(f"epoch {iter} / {epochs} => {print_loss_avg}")
