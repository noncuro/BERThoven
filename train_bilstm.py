import time

import torch
import torch.nn as nn
from torch import optim


class TrainerBiLSTM:
    """Class responsible for training the Bi-LSTM architecture
    """

    def __init__(
        self, encoder, decoder, device, max_length, loss_function=nn.MSELoss()
    ):
        """
        encoder: EncoderRNN type object. The encoder model to train
        decoder: AttnDecoderRNN type object. The decoder model to train
        device: The device to use. Either "cuda" or "cpu"
        max_length: Length of the largest sentence on the dataset
        loss_function: Loss function to use for training
        """
        self.device = device
        self.max_length = max_length
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.loss_function = loss_function

    def train_once(
        self, src_tensor, mt_tensor, score, encoder_optimizer, decoder_optimizer
    ):
        """Trains the models with one batch
        src_tensor: torch.Tensor representing indices of the words in the source language
        mt_tensor: torch.Tensor representing indices of the words in the translated language
        score: Score of the translation
        encoder_optimizer: The optimizer used to update the encoder's weights
        decoder_optimizer: The optimizer used to update the decoder's weights
        """
        self.encoder.train()  # Set encoder to training mode
        self.decoder.train()  # Set decoder to training mode
        encoder_hidden = self.encoder.init_hidden()  # Set the encoder's initial state

        encoder_optimizer.zero_grad()  # Clean any extraneous gradients left for the encoder
        decoder_optimizer.zero_grad()  # Clean any extraneous gradients left for the decoder

        src_length = src_tensor.size(
            0
        )  # Length of the sentences in the src_tensor batch
        mt_length = mt_tensor.size(0)  # Length of the sentences in the mt_tensor batch

        # Prepare a matrix to hold the encoder outputs (they'll be used in the attention mechanism)
        encoder_outputs = torch.zeros(
            self.max_length, self.encoder.hidden_size, device=self.device
        )

        # Iterate through every token in the source sentences batch
        for i in range(src_length):
            # Pass the tokens through the encoder
            encoder_output, encoder_hidden = self.encoder(src_tensor[i], encoder_hidden)

            # Store the output in the encoder_output matrix (to later use for attention)
            encoder_outputs[i] = encoder_output[0, 0]

        # Set the decoder's initial state to the encoder's final state (context vector)
        decoder_hidden = encoder_hidden

        # Iterate through every token in the translation sentences batch
        for i in range(mt_length):
            # Pass translation tokens and saved encoder_states through the encoder with attention mechanism
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                mt_tensor[i], decoder_hidden, encoder_outputs
            )

        # Use the last decoder output as the final score prediction
        # Calculate the loss based on the predicted and actual scores
        loss = self.loss_function(decoder_output, score)

        # Calculate gradients
        loss.backward()

        # Update models' weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Return the normalized loss for logging purposes
        return loss.item() / mt_length

    def train(self, epochs, print_every=1000, learning_rate=0.01):
        print_loss_total = 0  # Reset every print_every

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)
        training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(epochs)]

        for iter in range(1, epochs + 1):
            training_pair = training_pairs[iter - 1]
            src_tensor = training_pair[0]
            mt_tensor = training_pair[1]
            score = training_pair[2]

            loss = self.train_once(
                src_tensor, mt_tensor, score, encoder_optimizer, decoder_optimizer
            )
            print_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print(f"epoch {iter} / {epochs} => {print_loss_avg}")

    def predict(self, src_tensor, mt_tensor):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            encoder_hidden = self.encoder.init_hidden()
            src_length = src_tensor.size(0)
            mt_length = mt_tensor.size(0)

            encoder_outputs = torch.zeros(
                self.max_length, self.encoder.hidden_size, device=self.device
            )

            for i in range(src_length):
                encoder_output, encoder_hidden = self.encoder(
                    src_tensor[i], encoder_hidden
                )
                encoder_outputs[i] = encoder_output[0, 0]

            decoder_hidden = encoder_hidden
            for i in range(mt_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    mt_tensor[i], decoder_hidden, encoder_outputs
                )

            return decoder_output
