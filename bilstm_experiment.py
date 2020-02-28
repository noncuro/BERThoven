import torch

import comparer_model
import utils
from train_bilstm import TrainerBiLSTM

use_gpu = True

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Loading datasets")
train_df = utils.import_file("train")
dev_df = utils.import_file("dev")
test_df = utils.import_file("test")

print("Building tokenizer")
tokenizer = utils.Tokenizer()

batch_size = 32
hidden_size = 128
max_length = 200
epochs = 10
dropout_p = 0.1

print("Creating training dataloader")
dataloader_train = utils.get_data_loader_bilstm(
    train_df, tokenizer, batch_size=batch_size
)
print("Creating dev dataloader")
dataloader_dev = utils.get_data_loader_bilstm(dev_df, tokenizer, batch_size=batch_size)
print("Creating test dataloader")
dataloader_test = utils.get_data_loader_bilstm(
    test_df, tokenizer, batch_size=batch_size, test=True
)

print("Building encoder")
encoder = comparer_model.EncoderRNN(tokenizer.vocab_size, hidden_size)
print("Building decoder")
decoder = comparer_model.AttnDecoderRNN(
    tokenizer.vocab_size, hidden_size, max_length, dropout_p=dropout_p
)

print("Creating Trainer")
trainer = TrainerBiLSTM(encoder, decoder, batch_size, device, max_length)
print("Starting training")
trainer.train(dataloader_train, epochs)
