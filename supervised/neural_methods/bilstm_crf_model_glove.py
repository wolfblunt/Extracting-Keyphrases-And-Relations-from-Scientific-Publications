import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
torch.manual_seed(1)

class BiLSTMCRF(nn.Module):
    def __init__(self, sent_vocab, tag_vocab, dropout_rate, embed_size, hidden_size, embeds):
        super(BiLSTMCRF, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.sent_vocab = sent_vocab
        self.tag_vocab = tag_vocab
        self.embeds = embeds

        self.embedding = nn.Embedding(len(sent_vocab), embed_size)
        self.embedding.load_state_dict({'weight': embeds})

        self.hidden2emit_score = nn.Linear(hidden_size * 2, len(self.tag_vocab))
        self.transition = nn.Parameter(torch.randn(len(self.tag_vocab), len(self.tag_vocab)))
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, sentences, tags, sen_lengths):
        mask = (sentences != self.sent_vocab['<PAD>']).to(self.device)
        sentences = sentences.transpose(0, 1)
        sentences = self.embedding(sentences)
        emit_score = self.encode(sentences, sen_lengths)
        loss = self.calculate_loss(tags, mask, emit_score)
        return loss

    def encode(self, sentences, sent_lengths):
        padded_sentences = pack_padded_sequence(sentences, sent_lengths)
        hidden_states, _ = self.encoder(padded_sentences)
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True)
        emit_score = self.hidden2emit_score(hidden_states)
        emit_score = self.dropout(emit_score)
        return emit_score

    def calculate_loss(self, tags, mask, emit_score):
        _, sent_len = tags.shape
        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)

        d = torch.unsqueeze(emit_score[:, 0], dim=1)
        for i in range(1, sent_len):
            n_unfinished = mask[:, i].sum()
            d_uf = d[:n_unfinished]
            emit_and_transition = emit_score[:n_unfinished, i].unsqueeze(dim=1) + self.transition
            log_sum = d_uf.transpose(1, 2) + emit_and_transition
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)
            log_sum = log_sum - max_v
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)
        max_d = d.max(dim=-1)[0]
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)
        llk = total_score - d
        loss = -llk
        return loss

    def predict(self, sentences, sen_lengths):
        batch_size = sentences.shape[0]
        mask = (sentences != self.sent_vocab['<PAD>'])
        sentences = sentences.transpose(0, 1)
        sentences = self.embedding(sentences)
        emit_score = self.encode(sentences, sen_lengths)
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size
        d = torch.unsqueeze(emit_score[:, 0], dim=1)
        for i in range(1, sen_lengths[0]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[:n_unfinished]
            emit_and_transition = self.transition + emit_score[:n_unfinished, i].unsqueeze(dim=1)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()
            tags[:n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)
        _, max_idx = torch.max(d, dim=1)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags

    def save(self, filepath):
        params = {
            'sent_vocab': self.sent_vocab,
            'tag_vocab': self.tag_vocab,
            'args': dict(dropout_rate=self.dropout_rate, embed_size=self.embed_size, hidden_size=self.hidden_size,
                         embeds=self.embeds),
            'state_dict': self.state_dict()
        }
        torch.save(params, filepath)

    def load(filepath, device_to_load):
        params = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = BiLSTMCRF(params['sent_vocab'], params['tag_vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        model.to(device_to_load)
        return model
