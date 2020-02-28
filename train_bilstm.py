import torch
import torch.nn as nn
from torch import optim
from tokenizer import FullTokenizer
from torch.utils.data import Dataset
from utils import pad
from utils import prepro_df


class Tokenizer:
    def __init__(self, vocab_file="vocab.txt", do_lower_case=True):
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab_size = len(f.readlines())
        self.tk = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def tokenize(self, text):
        return self.tk.tokenize(text)

    def batch_tokenize(self, texts):
        return [self.tokenize(text) for text in texts]

    def convert_tokens_to_ids(self, tokens):
        return self.tk.convert_tokens_to_ids(tokens)

    def batch_convert_tokens_to_ids(self, texts):
        return [self.convert_tokens_to_ids(tokens) for tokens in texts]


class BiLSTMDataset(Dataset):
    def __init__(self, dataframe, _tokenizer, test=False):
        self.samples = []
        self.test = test
        src, mt = get_tokenized_one_way(dataframe, _tokenizer)
        x1, _ = pad(src)
        x2, _ = pad(mt)
        for i, _ in enumerate(range(len(x1))):
            if self.test:
                self.samples.append((x1[i], x2[i]))
            else:
                self.samples.append((x1[i], x2[i], dataframe.iloc[i].scores))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]


class TrainerBiLSTM:
    """Class responsible for training the Bi-LSTM architecture
    """

    def __init__(
        self,
        encoder,
        decoder,
        batch_size,
        device,
        max_length,
        loss_function=nn.MSELoss(),
    ):
        """
        encoder: EncoderRNN type object. The encoder model to train
        decoder: AttnDecoderRNN type object. The decoder model to train
        device: The device to use. Either "cuda" or "cpu"
        max_length: Length of the largest sentence on the dataset
        loss_function: Loss function to use for training
        """
        print(1)
        self.batch_size = batch_size
        print(2)
        self.device = device

        print(3)
        self.max_length = max_length

        print(4)
        self.encoder = encoder.to(device)

        print(5)
        self.decoder = decoder.to(device)

        print(6)
        self.loss_function = loss_function
        print(7)

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
        encoder_hidden = self.encoder.init_hidden(
            self.batch_size
        )  # Set the encoder's initial state

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
            print("=>", encoder_hidden.shape)
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

    def train(self, dataloader, epochs, learning_rate=0.01):
        print_loss_total = 0  # Reset loss count for logging purposes

        # Define the encoder optimizer
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)

        # Define the decoder optimizer
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        for e in range(epochs):
            for i, (src, mt, score) in enumerate(dataloader):
                src = src.to(device=self.device, dtype=torch.long).T
                mt = mt.to(device=self.device, dtype=torch.long).T
                score = score.to(device=self.device, dtype=torch.long)

                loss = self.train_once(
                    src, mt, score, encoder_optimizer, decoder_optimizer
                )
                print_loss_total += loss

            print_loss_avg = print_loss_total / i
            print_loss_total = 0
            print(f"epoch {e} / {epochs} => {print_loss_avg}")

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


def get_tokenized_one_way(dataframe, _tokenizer):
    input1 = (
        dataframe.apply(lambda a: a.src, axis=1)
        .apply(lambda a: _tokenizer.tokenize(a))
        .apply(lambda a: _tokenizer.convert_tokens_to_ids(a))
    )

    input2 = (
        dataframe.apply(lambda a: a.mt, axis=1)
        .apply(lambda a: _tokenizer.tokenize(a))
        .apply(lambda a: _tokenizer.convert_tokens_to_ids(a))
    )
    return input1, input2


def get_data_loader_bilstm(
    dataframe, _tokenizer, batch_size=32, test=False, preprocessor=None, fit=False
):
    dataframe = prepro_df(dataframe, preprocessor, fit)
    ds = BiLSTMDataset(dataframe, _tokenizer, test=test)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=(not test))
