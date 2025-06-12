import torch
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid, Amazon, Actor, WebKB, WikipediaNetwork
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn.functional as F
import os
import json
import warnings

warnings.filterwarnings('ignore')


def KL(alpha, c):
    beta = torch.ones((1, c))
    if torch.cuda.is_available():
        beta = beta.cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def random_drop(features, drop_rate, training):
    n = features.shape[0]
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    if training:
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        if torch.cuda.is_available():
            features = masks.cuda() * features
        else:
            features = masks * features
    else:
        features = features * (1. - drop_rate)
    return features


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if torch.cuda.is_available():
        features = torch.FloatTensor(features).cuda()
    return features


def Norm(x, min=0):
    x = x.detach().cpu().numpy()
    if min == 0:
        scaler = MinMaxScaler((0, 1))
    else:
        scaler = MinMaxScaler((-1, 1))
    norm_x = torch.tensor(scaler.fit_transform(x))
    if torch.cuda.is_available():
        norm_x = norm_x.cuda()

    return norm_x


def MsgPropagation(data, norm_A, args):
    K = args.num_hops
    X_list = []
    if args.dataset in ['Actor', 'chameleon']:
        X_list.append(preprocess_features(data.cpu().numpy()))
    else:
        X_list.append(Norm(data))
    for _ in range(K):
        X_list.append(torch.spmm(norm_A, X_list[-1]))
    return X_list


def get_dissonance(alpha):
    evidence = alpha - 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    belief = evidence / S
    belief_k = belief.unsqueeze(-1)
    belief_j = belief.unsqueeze(1)
    balances = 1 - torch.abs(belief_k - belief_j) / (belief_k + belief_j + 1e-7)
    zero_diag = torch.ones_like(balances[0])
    zero_diag.fill_diagonal_(0)
    balances *= zero_diag.unsqueeze(0)
    diss_numerator = (belief.unsqueeze(1) * balances).sum(dim=-1)
    diss_denominator = belief.sum(dim=-1, keepdim=True) - belief + 1e-7
    diss = (belief * diss_numerator / diss_denominator).sum(dim=-1)
    return diss


def ce_loss(p, alpha, c):
    S = torch.sum(alpha, dim=1, keepdim=True)
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    return A


def reg_loss(p, E, c, kl, dis):
    alpha = E + 1.0
    label = F.one_hot(p, num_classes=c)
    alp = E * (1 - label) + 1
    C = kl * KL(alp, c)
    D = dis * get_dissonance(alpha).unsqueeze(-1)
    return C + D


def normalize_adj(adj, flag=True):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if flag:
        adj = adj + sp.eye(adj.shape[0])
    else:
        adj = adj
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def get_dataset(ds, seed):
    if ds in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./data/Planetoid/', name=ds)
        dataset.x = dataset[0].x
        dataset.y = dataset[0].y
        dataset.edge_index = dataset[0].edge_index
        dataset.train_mask = dataset[0].train_mask
        dataset.val_mask = dataset[0].val_mask
        dataset.test_mask = dataset[0].test_mask
    elif ds in ['Computers', 'Photo']:
        dataset = Amazon(root='./data/Amazon/', name=ds)
        dataset = set_train_val_test_split(seed, dataset)
    elif ds in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root='./data/', name=ds)
        dataset = set_train_val_test_split(seed, dataset)
    elif ds in ['Actor']:
        dataset = Actor(root='./data/Actor')
        dataset = set_train_val_test_split(seed, dataset)
    elif ds in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root='./data/', name=ds)
        dataset = set_train_val_test_split(seed, dataset)
    else:
        raise Exception('Unknown dataset.')
    data, edge_index, target = dataset.x, dataset.edge_index, dataset.y
    num_nodes = data.shape[0]
    adj = sp.coo_matrix((torch.ones(edge_index.shape[1]), edge_index), shape=(num_nodes, num_nodes))
    if ds in ['Actor', 'chameleon', 'squirrel']:
        adj = normalize_adj(adj, False)
    else:
        adj = normalize_adj(adj)
    adj = torch.FloatTensor(adj.toarray())
    if torch.cuda.is_available():
        adj = adj.cuda()
        data = data.cuda()
    dataset.adj = adj
    dataset.target = target
    dataset.x = data
    return dataset


def set_train_val_test_split(seed: int, data: Data, train_per_class=20, val_per_class=30) -> Data:
    rnd_state = np.random.RandomState(seed)

    data.y = data[0].y
    data.x = data[0].x
    data.edge_index = data[0].edge_index
    num_nodes = data.y.shape[0]

    num_train_per_class = train_per_class
    num_val_per_class = val_per_class

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)

    perm = rnd_state.permutation(num_nodes)

    train_mask_per_class = []
    val_mask_per_class = []
    for c in range(data.y.max() + 1):
        class_nodes = perm[np.where(data.y[perm] == c)[0]]
        train_mask_per_class.extend(class_nodes[:num_train_per_class])
        val_mask_per_class.extend(class_nodes[num_train_per_class:num_train_per_class + num_val_per_class])

    test_mask[:] = True
    test_mask[torch.cat((torch.tensor(train_mask_per_class), torch.tensor(val_mask_per_class)))] = False

    train_mask[train_mask_per_class] = True
    val_mask[val_mask_per_class] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def load_best_params(dataset, file_path='./params/best_params.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        if str(dataset) in data:
            return data[str(dataset)]
    return None
