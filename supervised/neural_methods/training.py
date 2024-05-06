import torch
import torch.nn as nn
import itertools
import numpy as np
from tqdm import tqdm
import itertools
from sklearn.metrics import classification_report, accuracy_score
import os
import sys
import gensim.downloader as api
from bilstm_crf_model_glove import BiLSTMCRF
from bilstm_crf_model_naive import BiLSTMCRFNaive
from data_processing import read_corpus, generate_train_dev_dataset, init_vocabs
from data_processing import pad, batch_iter, calculate_dev_loss, words2indices, create_vocab

torch.manual_seed(1)
model_path = "./model/"
datasetPath = './datasetPath/'
wv = api.load('glove-wiki-gigaword-300')


def train():
    dataset = sys.argv[1]
    dataset_name = dataset.split('_')[0]
    model_name = dataset.split('_')[1]
    print("dataset_name: ", dataset_name)
    sent_vocab, tag_vocab = init_vocabs(dataset_name)
    v = sent_vocab.get_itos()
    matrix_len = len(v)
    weights_matrix = np.zeros((matrix_len, 300))

    for i, word in enumerate(v):
        try:
            weights_matrix[i] = wv[word]
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(300,))
    
    train_data, dev_data = generate_train_dev_dataset(datasetPath+f'{dataset_name}_train.txt', sent_vocab, tag_vocab)

    max_epoch = 15
    valid_freq = 20
    madelPath  = './model/'
    model_save_path =madelPath +f'./{dataset}_model.pth'
    optimizer_save_path = f'./{dataset}_optim.pth'
    min_dev_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dropout_rate = 0.5
    embed_size = 300
    hidden_size = 300
    batch_size = 128
    max_clip_norm = 5.0
    lr_decay = 0.5

    if model_name == 'glove':
        model = BiLSTMCRF(sent_vocab, tag_vocab, dropout_rate, embed_size, hidden_size,
                          torch.tensor(weights_matrix)).to(device)
    else:
        model = BiLSTMCRFNaive(sent_vocab, tag_vocab, dropout_rate, embed_size, hidden_size).to(device)

    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, 0, 0.01)
        else:
            nn.init.constant_(param.data, 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_iter = 0  # train iter num

    print('Training...')
    for epoch in tqdm(range(max_epoch)):
        for sentences, tags in batch_iter(train_data, batch_size=batch_size):
            train_iter += 1
            sentences, sent_lengths = pad(sentences, sent_vocab['<PAD>'], device)
            tags, _ = pad(tags, tag_vocab['<PAD>'], device)

            # back propagation
            optimizer.zero_grad()
            batch_loss = model(sentences, tags, sent_lengths)
            loss = batch_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_clip_norm)
            optimizer.step()

            if train_iter % valid_freq == 0:
                dev_loss = calculate_dev_loss(model, dev_data, 64, sent_vocab, tag_vocab, device)
                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    model.save(model_save_path)
                    torch.save(optimizer.state_dict(), optimizer_save_path)
                else:
                    lr = optimizer.param_groups[0]['lr'] * lr_decay
                    model = BiLSTMCRF.load(model_save_path, device)
                    optimizer.load_state_dict(torch.load(optimizer_save_path))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr


def test():
    dataset = sys.argv[1]
    dataset_name, model_name = dataset.split('_')
    # print(f'Test {dataset_name}_test.txt')
    print("dataset_name: ", dataset_name, ", model_name: ", model_name)
    try:
        sent_vocab, tag_vocab = init_vocabs(dataset_name)
    except Exception as e:
        print("test message mai error hai : ", e)
    sentences, tags = read_corpus(datasetPath+f'{dataset_name}_test.txt')
    sentences = words2indices(sentences, sent_vocab)
    tags = words2indices(tags, tag_vocab)
    test_data = list(zip(sentences, tags))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'glove':
        model = BiLSTMCRF.load(model_path + f'model_{dataset}.pth', device)
    else:
        model = BiLSTMCRFNaive.load(model_path + f'model_{dataset}.pth', device)
    batch_size = 128

    predictions = []
    model.eval()

    y_true = []
    with torch.no_grad():
        for sentences, tags in batch_iter(test_data, batch_size=int(batch_size), shuffle=False):
            padded_sentences, sent_lengths = pad(sentences, sent_vocab['<PAD>'], device)
            predicted_tags = model.predict(padded_sentences, sent_lengths)

            predictions.append(predicted_tags)
            y_true.append(tags)

    return predictions, y_true


if __name__ == '__main__':
    dataset = sys.argv[1]
    dataset_name = dataset.split('_')[0]
    print("dataset_name: ", dataset_name)
    if os.path.exists(model_path + f'model_{dataset}.pth'):
        pass
    else:
        train()

    y_hat, y_true = test()

    y_hat = list(itertools.chain(*list(itertools.chain(*y_hat))))
    y_true = list(itertools.chain(*list(itertools.chain(*y_true))))

    print(f"For dataset {dataset}")
    print(accuracy_score(y_hat, y_true))
    print(classification_report(y_hat, y_true, zero_division=0.0))

    sent, tags = init_vocabs(dataset_name)
    print(tags.get_stoi())
