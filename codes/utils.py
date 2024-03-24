import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix

import pickle as pkl
from tqdm import tqdm


def pkl_load(path):
    with open(path,'rb') as f:
        return pkl.load(f)



def pad_seq(seq:list[int], obj_len, pad_token):
    """pad the sequence with pad_token"""
    if len(seq) > obj_len:
        return seq[:obj_len]
    else:
        padded_seq = seq + [pad_token]*(obj_len-len(seq))
        return padded_seq



def get_loader(data,batch_size,train=True):
    """get the iterator of data"""
    """data: a tuple (X,y), X and y should be tensor"""
    """train: shuffle the dataset before each epoch"""

    dataset = TensorDataset(*data)
    return DataLoader(dataset,batch_size=batch_size,shuffle=train)



def load_data(obj_len, batch_size=64):
    # load the vocabulary
    vocab_path = "../processed_data/remove-stopwords-punct-25000.vocab"
    vocab = torch.load(vocab_path)

    # load training, validation and test sets
    train_path = "../processed_data/train.pickle"
    val_path = "../processed_data/val.pickle"
    test_path = "../processed_data/test.pickle"

    X_train,y_train = pkl_load(train_path)
    X_val,y_val = pkl_load(val_path)
    X_test,y_test = pkl_load(test_path)

    # load embedding matrix
    word2vec_path = "../processed_data/word2vec.pickle"
    glove_path = "../processed_data/glove.pickle"

    embed_matrix_word2vec = pkl_load(word2vec_path)
    embed_matrix_glove = pkl_load(glove_path)

    # padding
    pad_token = vocab['<PAD>']
    X_train = torch.tensor([pad_seq(text,obj_len,pad_token) for text in X_train])
    X_val = torch.tensor([pad_seq(text,obj_len,pad_token) for text in X_val])
    X_test = torch.tensor([pad_seq(text,obj_len,pad_token) for text in X_test])

    # turn ys into tensor
    # y is in {1,2,3,4}, convert the labels to {0,1,2,3} so that data can be processed by torch
    y_train = torch.tensor(y_train) - 1
    y_val = torch.tensor(y_val) - 1
    y_test = torch.tensor(y_test) - 1

    # get the data loader
    train_iter = get_loader((X_train,y_train),batch_size,train=True)
    val_iter = get_loader((X_val,y_val),batch_size,train=False)
    test_iter = get_loader((X_test,y_test),batch_size,train=False)

    res = {'train_iter': train_iter,
           'val_iter': val_iter,
           'test_iter': test_iter,
           'word2vec': torch.tensor(embed_matrix_word2vec),
           'glove': torch.tensor(embed_matrix_glove)}
    return res



def calculate_metric(net, data_loader, device):
    """calculate macro F1-score"""
    net.eval()
    true_labels = []
    predict_labels = []
    
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            output = net(X)
            _, y_pred = torch.max(output.data, 1)
            
            true_labels.extend(y.cpu().numpy())
            predict_labels.extend(y_pred.cpu().numpy())
    
    # Calculate metrics
    f1 = f1_score(true_labels, predict_labels, average='macro')
    
    return f1



def get_cm(net, data_loader, device):
    """get confusion matrix"""
    net.eval()
    true_labels = []
    predict_labels = []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            output = net(X)
            _, y_pred = torch.max(output.data, 1)
            
            true_labels.extend(y.cpu().numpy())
            predict_labels.extend(y_pred.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(true_labels, predict_labels, normalize="true")
    return cm


def train_batch(net, X, y, loss, optimizer, device):
    """train the model on a mini batch"""
    X = X.to(device)
    y = y.to(device)

    net.train()
    optimizer.zero_grad()
    y_pred = net(X)
    l = loss(y_pred, y)
    l.backward()
    optimizer.step()

    return l



def plot_training_process(f1_scores_train,f1_scores_val):
    sns.set_style('whitegrid')
    xs = np.arange(1, len(f1_scores_train)+1)

    plt.plot(xs, f1_scores_train, label='Train F1 Score', marker='o')
    plt.plot(xs, f1_scores_val, label='Validation F1 Score', marker='x')

    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Train vs Validation F1 Score')

    plt.legend()
    plt.grid(True)



def train(net, train_iter, val_iter, num_epochs, lr=1e-3, verbose=True):
    """full training process"""
    """verbose: print training process if True"""

    # check cuda
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    f1_scores_train = []
    f1_scores_val = []

    # set loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    with tqdm(total=num_epochs) as progress_bar:
        for epoch in range(num_epochs):
            for i, (X, y) in enumerate(train_iter):
                train_batch(net, X, y, loss, optimizer, device)

            # update training process
            train_f1 = calculate_metric(net, train_iter, device)
            val_f1 = calculate_metric(net, val_iter, device)
            f1_scores_train.append(train_f1)
            f1_scores_val.append(val_f1)

            if verbose: #print training process
                print(f'epoch {epoch}   train_f1: {train_f1:.3f}, val_f1: {val_f1:.3f}')
                print()

            progress_bar.update(1)  #update progress bar

    plot_training_process(f1_scores_train,f1_scores_val)
            


