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
        score,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        loss_function,
        max_length,
    ):
        encoder.train()
        decoder.train()
        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        src_length = src_tensor.size(0)
        mt_length = mt_tensor.size(0)

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=self.device
        )

        for i in range(src_length):
            encoder_output, encoder_hidden = encoder(src_tensor[i], encoder_hidden)
            encoder_outputs[i] = encoder_output[0, 0]

        decoder_hidden = encoder_hidden
        for i in range(mt_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                mt_tensor[i], decoder_hidden, encoder_outputs
            )

        loss = loss_function(decoder_output, score)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / mt_length

    def train(
        self, encoder, decoder, epochs, print_every=1000, learning_rate=0.01
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

            loss = self.train_once(
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

    def evaluate(self, encoder, decoder, sentence, max_length):
        with torch.no_grad():
            input_tensor = tensorFromSentence(input_lang, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()

            encoder_outputs = torch.zeros(
                max_length, encoder.hidden_size, device=device
            )

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden
                )
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append("<EOS>")
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[: di + 1]

    def predict(
        self,
        src_tensor,
        mt_tensor,
        score,
        encoder,
        decoder,
        max_length,
    ):
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            encoder_hidden = encoder.initHidden()
            src_length = src_tensor.size(0)
            mt_length = mt_tensor.size(0)

            encoder_outputs = torch.zeros(
                max_length, encoder.hidden_size, device=self.device
            )

            for i in range(src_length):
                encoder_output, encoder_hidden = encoder(src_tensor[i], encoder_hidden)
                encoder_outputs[i] = encoder_output[0, 0]

            decoder_hidden = encoder_hidden
            for i in range(mt_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    mt_tensor[i], decoder_hidden, encoder_outputs
                )

            return decoder_output
