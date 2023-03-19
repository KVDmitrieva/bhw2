import torch

from torch import nn
from torch.utils.data import DataLoader

from model import LanguageModel
from dataset import TextDataset, DatasetAdapter
from train import train


BATCH_SIZE = 128
TRAIN_SIZE = 1.
VOCAB_SIZE = 7000

TRAIN_PATH = './bhw2-data/data/train.de-en.'
VALID_PATH = './bhw2-data/data/val.de-en.'
FILE_IN = './bhw2-data/data/test1.de-en.de'
FILE_OUT = 'test1.de-en.en'

NUM_EPOCHS = 5

EMBED_SIZE = 512
HIDDEN_SIZE = 512
RNN_LAYERS = 1

LR = 1e-3
GAMMA = 0.1
MILESTONE = [3, 7, 10, 13]


def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)


def make_prediction_file(model, source_file, output_file):
    with open(source_file) as file:
        texts = file.readlines()

    with open(output_file, 'w') as file:
        for line in texts:
            file.write(model.inference(line, 2) + '\n')


def main():
    # tokenize text, make loaders
    train_de_files = [TRAIN_PATH + 'de', VALID_PATH + 'de', FILE_IN]
    train_en_files = [TRAIN_PATH + 'en', VALID_PATH + 'en']

    train_de = TextDataset(data_file=TRAIN_PATH + 'de', train_files=train_de_files,
                           sp_model_prefix='de', vocab_size=VOCAB_SIZE, train_size=TRAIN_SIZE)

    train_en = TextDataset(data_file=TRAIN_PATH + 'en', train_files=train_en_files,
                           sp_model_prefix='en', vocab_size=VOCAB_SIZE, train_size=TRAIN_SIZE)

    valid_de = TextDataset(data_file=VALID_PATH + 'de', train_files=train_de_files, sp_model_prefix='de')
    valid_en = TextDataset(data_file=VALID_PATH + 'en', train_files=train_en_files, sp_model_prefix='en')

    train_set = DatasetAdapter(train_de, train_en)
    valid_set = DatasetAdapter(valid_de, valid_en)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)

    # train model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    rnn_model = LanguageModel(train_de, train_en, rnn_type=nn.GRU, embed_size=EMBED_SIZE,
                              hidden_size=HIDDEN_SIZE, rnn_layers=RNN_LAYERS).to(device)
    rnn_model.apply(init_weights)
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONE, gamma=GAMMA)

    train(rnn_model, optimizer, scheduler, train_loader, valid_loader, NUM_EPOCHS)

    # make prediction
    file1 = FILE_IN
    file2 = FILE_OUT

    make_prediction_file(rnn_model, file1, file2)


if __name__ == '__main__':
    main()
