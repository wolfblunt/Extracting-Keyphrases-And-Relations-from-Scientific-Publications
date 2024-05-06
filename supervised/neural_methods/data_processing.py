import torch
import random
from torchtext.vocab import vocab
import itertools
from collections import OrderedDict
import itertools

torch.manual_seed(1)
random.seed(99)
datasetPath = './datasetPath/'

def read_corpus(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = file.read()

    data = data.split('\n')
    X = []
    y = []

    for data_line in data:
        data_line = data_line.split('\t')
        if data_line[0] != '':
            X.append(data_line[0])
        if data_line[-1] != '':
            y.append(data_line[-1])

    X = [word.lower() for word in X]

    newX = []
    newY = []

    sentence = []
    tags = []

    for i in range(0, len(X)):
        if X[i] == '.':
            tags.append(y[i])
            sentence.append(X[i])
            newX.append(sentence)
            newY.append(tags)
            sentence = []
            tags = []
        else:
            sentence.append(X[i])
            tags.append(y[i])

    newX = [' '.join(x) for x in newX]
    newY = [' '.join(y) for y in newY]

    return newX, newY


def generate_train_dev_dataset(filepath, sent_vocab, tag_vocab, train_proportion=0.8):
    sentences, tags = read_corpus(filepath)

    sentences = words2indices(sentences, sent_vocab)
    tags = words2indices(tags, tag_vocab)

    data = list(zip(sentences, tags))
    random.shuffle(data)
    n_train = int(len(data) * train_proportion)
    train_data, dev_data = data[:n_train], data[n_train:]
    return train_data, dev_data


def words2indices(origin, vocab):
    result = [[vocab[w] for w in sent.split()] for sent in origin]
    return result


def pad(data, padded_token, device):
    lengths = [len(sent) for sent in data]
    max_len = lengths[0]
    padded_data = []
    for s in data:
        padded_data.append(s + [padded_token] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths


def create_vocab(tokens):
    try:
        tokens = [t.split() for t in tokens]
        tokens = list(itertools.chain(*tokens))
        tokens = list(set(tokens))
        v = OrderedDict()
        for i in tokens:
            v[i] = len(v) + 2
        vocabulary = vocab(v, specials=['<UNK>', '<PAD>'])
        vocabulary.set_default_index(vocabulary['<UNK>'])
        return vocabulary
    except Exception as e:
        print("create_vocab mai error aa raha hai", e)


def init_vocabs(dataset):
    sent, tags = read_corpus(datasetPath + f'{dataset}_train.txt')
    sent_vocab = create_vocab(sent)
    tag_vocab = create_vocab(tags)
    return sent_vocab, tag_vocab


def batch_iter(data, batch_size=32, shuffle=True):
    data_size = len(data)
    indices = list(range(data_size))
    if shuffle:
        random.shuffle(indices)
    batch_num = (data_size + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sentences = [x[0] for x in batch]
        tags = [x[1] for x in batch]
        yield sentences, tags


def calculate_dev_loss(model, dev_data, batch_size, sent_vocab, tag_vocab, device):
    is_training = model.training
    model.eval()
    loss, n_sentences = 0, 0
    with torch.no_grad():
        for sentences, tags in batch_iter(dev_data, batch_size, shuffle=False):
            sentences, sent_lengths = pad(sentences, sent_vocab['<PAD>'], device)
            tags, _ = pad(tags, tag_vocab['<PAD>'], device)
            batch_loss = model(sentences, tags, sent_lengths)
            loss += batch_loss.sum().item()
            n_sentences += len(sentences)
    model.train(is_training)
    return loss / n_sentences
