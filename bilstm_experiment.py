import torch

import comparer_model
import utils
from train_bilstm import TrainerBiLSTM

use_gpu = True

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_df = utils.import_file("train")
dev_df = utils.import_file("dev")
test_df = utils.import_file("test")

dataloader_train = utils.get_data_loader_bilstm(train_df, batch_size=32)
dataloader_dev = utils.get_data_loader_bilstm(dev_df, batch_size=32)
dataloader_test = utils.get_data_loader_bilstm(test_df, batch_size=32, test=True)

hidden_size = 128
max_length = 200
epochs = 10
dropout_p = 0.1

encoder = comparer_model.EncoderRNN(vocab_size, hidden_size)
decoder = comparer_model.AttnDecoderRNN(
    vocab_size, hidden_size, max_length, dropout_p=dropout_p
)

trainer = TrainerBiLSTM(encoder, decoder, device, max_length)
trainer.train(dataloader_train, epochs)
