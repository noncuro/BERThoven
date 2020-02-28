import comparer_model
from train_bilstm import TrainerBiLSTM
import torch
import utils

use_gpu = True

if use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_df = utils.import_file("train")
dev_df = utils.import_file("dev")
test_df = utils.import_file("test")

dataLoader_train = utils.get_data_loader(train_df, batch_size=32, fit=True)
dataLoader_dev = utils.get_data_loader(dev_df, batch_size=32)
dataLoader_test = utils.get_data_loader(test_df, batch_size=32, test=True)


hidden_size = 128
max_length = 200

encoder = comparer_model.EncoderRNN(vocab_size, hidden_size)
decoder = comparer_model.AttnDecoderRNN(vocab_size, hidden_size, max_length, dropout_p=0.1)

trainer = TrainerBiLSTM(encoder, decoder, device, max_length)
