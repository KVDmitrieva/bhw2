import os
import torch

from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset


"""
Взято из 3 маленького дз
"""


class TextDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42
    VAL_RATIO = 0.05

    def __init__(self, data_file, train_files, sp_model_prefix, vocab_size=2000, train_size=1.0,
                 normalization_rule_name='nmt_nfkc_cf', model_type='word', max_length=600):
        """
        Dataset with texts, supporting BPE tokenizer
        :param data_file: txt file containing texts
        :param train_files: txt files containing texts for tokenizer trainig
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece tokenizer model type
        :param max_length: maximal length of text in tokens
        """
        if not os.path.isfile(sp_model_prefix + '.model'):
            # train tokenizer if not trained yet
            SentencePieceTrainer.train(
                input=train_files, vocab_size=vocab_size,
                model_type=model_type, model_prefix=sp_model_prefix,
                normalization_rule_name=normalization_rule_name, pad_id=3
            )
        # load tokenizer from file
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + '.model')

        with open(data_file) as file:
            self.texts = file.readlines()

        self.texts = self.texts[:int(train_size * len(self.texts))]

        self.indices = self.sp_model.encode(self.texts)

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_model.pad_id(), self.sp_model.unk_id(), \
                self.sp_model.bos_id(), self.sp_model.eos_id()
        self.max_length = max_length
        self.vocab_size = self.sp_model.vocab_size()

    def text2ids(self, texts):
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return self.sp_model.encode(texts)

    def ids2text(self, ids):
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.sp_model.decode(ids)

    def __len__(self):
        """
        Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.indices)

    def __getitem__(self, item):
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """

        indices = [self.bos_id] + self.indices[item] + [self.eos_id]
        length = len(indices)

        padded = torch.full((self.max_length,), self.pad_id)
        padded[:length] = torch.tensor(indices)
        return padded, length


class DatasetAdapter(Dataset):
    def __init__(self, first_dataset, second_dataset):
        """
        Adapter for merging two datasets
        :param first_dataset: first TextDataset to merge
        :param second_dataset: second TextDataset to merge
        """
        assert len(first_dataset) == len(second_dataset), "Datasets must be of the same length!"

        self.first = first_dataset
        self.second = second_dataset

    def __len__(self):
        return len(self.first)

    def __getitem__(self, item):
        first_item, first_len = self.first[item]
        second_item, second_len = self.second[item]
        return first_item, first_len, second_item, second_len

    def ids2text(self, ids1, ids2):
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        return self.first.ids2text(ids1), self.second.ids2text(ids2)