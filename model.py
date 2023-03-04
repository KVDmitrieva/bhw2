import torch
import torch.nn.functional as F

from typing import Type, Tuple
from torch import nn

from dataset import TextDataset


class Encoder(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1, dropout: float = 0.3):
        """
        Module for encoding input sequence
        :param dataset: text data dataset (to extract vocab_size and pad_id)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.GRU)
        :param rnn_layers: number of layers in RNN
        :param dropout: probability of an element to be zeroed
        """
        super(Encoder, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(num_embeddings=dataset.vocab_size,
                                      embedding_dim=embed_size,
                                      padding_idx=dataset.pad_id
                                      )

        self.rnn = rnn_type(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=rnn_layers,
                            batch_first=True,
                            bidirectional=True
                            )

        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forward pass through the model and return rnn output and
        combined forward and reverse hidden state from last layer
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of rnn output of shape (batch_size, length, 2 * hidden_size)
        :return: FloatTensor of hidden state of shape (batch_size, hidden_size)
        """

        embeds = self.dropout(self.embedding(indices))

        packed_seq = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed_seq)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        hidden = torch.tanh(self.linear(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return output, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_size: int, dec_hid_size: int):
        """
        Attention module to process encoder output and current decoder hidden state
        :param enc_hid_size: dimensionality of encoder hidden state
        :param dec_hid_size: dimensionality of decoder hidden state
        """
        super().__init__()

        self.attn = nn.Linear((enc_hid_size * 2) + dec_hid_size, dec_hid_size)
        self.linear = nn.Linear(dec_hid_size, 1, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute Bahdanau attention (multi-layer perceptron)
        :param hidden: FloatTensor of hidden state of shape (batch_size, decoder_hidden_size)
        :param encoder_outputs: FloatTensor of encoder output of shape (batch_size, src_length, 2 * encoder_hidden_size)
        :param mask: BoolTensor of non-padd indices of src_input of shape (batch_size, max_src_len)
        :return: FloatTensor of attention of shape (batch_size, src_length)
        """

        src_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        attention = self.linear(energy).squeeze(2)

        attention = attention.masked_fill(mask[:, :attention.shape[1]] == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1, dropout=0.3):
        """
        Module for encoding input sequence
        :param dataset: text data dataset (to extract vocab_size and pad_id)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.GRU)
        :param rnn_layers: number of layers in RNN
        :param dropout: probability of an element to be zeroed
        """
        super(Decoder, self).__init__()

        self.attention = Attention(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(num_embeddings=dataset.vocab_size,
                                      embedding_dim=embed_size,
                                      padding_idx=dataset.pad_id
                                      )

        self.rnn = rnn_type(input_size=embed_size + 2 * hidden_size,
                            hidden_size=hidden_size,
                            #                             num_layers=rnn_layers,
                            batch_first=True
                            )

        self.linear = nn.Linear(4 * hidden_size, dataset.vocab_size)

    def forward(self, indices, hidden, encoder_outputs, mask) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token
        :param indices: LongTensor of encoded tokens of size (batch_size, )
        :param hidden: FloatTensor of hidden state of shape (batch_size, decoder_hidden_size)
        :param encoder_outputs: FloatTensor of encoder output of shape (batch_size, src_length, 2 * encoder_hidden_size)
        :param mask: BoolTensor of non-padd indices of src_input of shape (batch_size, max_src_len)
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        :return: FloatTensor of hidden state of shape (batch_size, decoder_hidden_size)
        """

        embeds = self.dropout(self.embedding(indices))

        a = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)

        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embeds, weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        embeds = embeds.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        logits = self.linear(torch.cat((output, weighted, embeds), dim=1))

        return logits, hidden.squeeze(0)


class LanguageModel(nn.Module):
    def __init__(self, dataset_src: TextDataset, dataset_trg: TextDataset, embed_size: int = 256,
                 hidden_size: int = 256, rnn_type: Type = nn.RNN, rnn_layers: int = 1, dropout: float = 0.3):
        """
        Model for text translation
        :param dataset_src: source text data dataset (to extract vocab_size and max_length)
        :param dataset_trg: source text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.GRU)
        :param rnn_layers: number of layers in RNN
        :param dropout: probability of an element to be zeroed
        """
        super(LanguageModel, self).__init__()

        # for inference mode
        self.dataset_de = dataset_src
        self.dataset_eng = dataset_trg
        self.max_length = dataset_trg.max_length

        self.encoder = Encoder(dataset_src, embed_size, hidden_size, rnn_type, rnn_layers, dropout)
        self.decoder = Decoder(dataset_trg, embed_size, hidden_size, rnn_type, rnn_layers, dropout)

    def create_mask(self, src):
        mask = (src != self.dataset_de.pad_id)
        return mask

    def sample_n_tokens(self, logits: torch.Tensor, num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n different tokens
        :param logits: FloatTensor of logits of shape (batch_size, trg_length, vocab_size)
        :param num: number of tokens to sample
        :return: FloatTensor with probabilities of chosen tokens of shape (batch_size, num)
        :return: LongTensor with tokens of shape(batch_size, num)
        """

        vals, inds = F.softmax(logits, dim=1).topk(k=num, dim=1)
        return vals, inds

    def forward(self, src_indices: torch.Tensor, src_lengths: torch.Tensor,
                trg_indices: torch.Tensor, trg_lengths: torch.Tensor,
                teacher_forcing_ratio: float = 0.) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token
        :param src_indices: LongTensor of encoded source tokens of size (batch_size, src_length)
        :param src_lengths: LongTensor of source lengths of size (batch_size, )
        :param trg_indices: LongTensor of encoded target tokens of size (batch_size, trg_length)
        :param trg_lengths: LongTensor of target lengths of size (batch_size, )
        :param teacher_forcing_ratio: probability of using target token as a decoder input
        :return: FloatTensor of logits of shape (batch_size, trg_length, vocab_size)
        """

        enc_out, hidden = self.encoder(src_indices, src_lengths)

        input_token = torch.reshape(trg_indices[:, 0], (-1, 1))
        mask = self.create_mask(src_indices)
        logits = None

        for i in range(1, trg_lengths.max()):
            logit, hidden = self.decoder(input_token, hidden, enc_out, mask)

            logit = logit.unsqueeze(1)
            logits = logit if logits is None else torch.cat([logits, logit], dim=1)

            _, next_token = self.sample_n_tokens(logit.squeeze(1), 1)

            teacher_forcing = torch.rand(1)[0] < teacher_forcing_ratio
            input_token = trg_indices[:, i] if teacher_forcing else next_token
            input_token = torch.reshape(input_token, (-1, 1))

        return logits

    def postprocess(self, src_text: str, trg_text: str) -> str:
        """
        Post-process for model translation
        :param src_text: source text
        :param trg_text: model translation
        :return: post-processed model translation
        """
        words = trg_text.split()

        # remove unknown tokens
        words = [word for word in words if word != '⁇']

        # remove duplicates
        if len(words) != 0:
            words = [words[0]] + [words[i] for i in range(1, len(words)) if words[i] != words[i - 1]]

        # fix punctuation
        if len(words) != 0 and words[-1] not in ['.', '?', '!', '"']:
            words += src_text[-2]

        if '"' not in src_text:
            words = [word for word in words if word != '"']

        trg_text = ' '.join(words)
        return trg_text

    @torch.inference_mode()
    def inference(self, src_input: str, beam_size: int = 2) -> str:
        """
        Translate given text (beam search)
        :param src_input: text to translate
        :param beam_size: number of alternatives to keep
        :return: translated text
        """
        self.eval()

        device = next(self.parameters()).device

        enc_input = [self.dataset_de.bos_id] + self.dataset_de.sp_model.encode(src_input) + [self.dataset_de.eos_id]
        len_inp = torch.tensor([len(enc_input)]).repeat(beam_size)

        enc_input = torch.tensor([enc_input]).repeat(beam_size, 1).to(device)
        enc_out, hidden = self.encoder(enc_input, len_inp)

        mask = self.create_mask(enc_input)

        input_tokens = torch.tensor([self.dataset_eng.bos_id]).unsqueeze(0).repeat(beam_size, 1).to(device)
        main_candidates = input_tokens

        logits, hidden = self.decoder(input_tokens, hidden, enc_out, mask)
        probs, new_tokens = self.sample_n_tokens(logits.squeeze(1), beam_size)

        main_probs = probs[0].reshape(-1)
        input_tokens = new_tokens[0].unsqueeze(1)
        main_candidates = torch.cat([main_candidates, input_tokens], dim=1)

        ended_seqs = []
        ended_seq_probs = torch.tensor([]).to(device)
        count = 0

        while main_candidates.shape[1] < 2 * len_inp[0]:
            count += 1
            logits, hidden = self.decoder(input_tokens, hidden, enc_out, mask)
            probs, new_tokens = self.sample_n_tokens(logits.squeeze(1), beam_size)

            best_probs, inds = probs.reshape(1, -1).topk(k=beam_size)
            inds = inds.reshape(-1)

            input_tokens = new_tokens[inds // beam_size, inds % beam_size].unsqueeze(1)
            main_candidates = torch.cat([main_candidates[inds // beam_size], input_tokens], dim=1)
            main_probs = main_probs[inds // beam_size] * best_probs.reshape(-1)
            main_probs *= 10

            hidden = hidden[inds // beam_size]

            if self.dataset_eng.eos_id in main_candidates[:, -1]:
                end_main = main_candidates[main_candidates[:, -1] == self.dataset_eng.eos_id]
                end_prob = main_probs[main_candidates[:, -1] == self.dataset_eng.eos_id]
                text = self.dataset_eng.ids2text(end_main.squeeze())

                ended_seqs += [text] if type(text) is str else text
                ended_seq_probs = torch.cat([ended_seq_probs, end_prob.reshape(-1)], dim=0)

        main_probs = torch.cat([main_probs, ended_seq_probs])
        text = self.dataset_eng.ids2text(main_candidates.squeeze())
        text += ended_seqs
        text = text[main_probs.argmax()]
        text = self.postprocess(src_input, text)
        return text
