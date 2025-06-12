import argparse
from utils import get_dataset, MsgPropagation, load_best_params, set_seeds
from train_test import train, test
import numpy as np


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='the train epochs')
    parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--hid-dim', type=int, default=32)
    parser.add_argument('--input-dim', type=int, default=16, help='the hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0., help='the dropout')
    parser.add_argument('--dataset', type=str, default='cora', help='the dataset')
    parser.add_argument('--num-class', type=int, default=0, help='the num_class')
    parser.add_argument('--weight-decay', type=float, default=0.02, help='the weight of decay')
    parser.add_argument('--num-hops', type=int, default=16)
    parser.add_argument('--patience-period', type=int, default=300)
    parser.add_argument('--dropnode-rate', type=float, default=0.)
    parser.add_argument('--input-droprate', type=float, default=0.)
    parser.add_argument('--val', type=bool, default=True)
    parser.add_argument('--kl', type=float, default=0.)
    parser.add_argument('--dis', type=float, default=0.)
    args = parser.parse_known_args()[0]
    return args


def main(data_name):
    args = parameter_parser()
    args.model_type = 'EFGNN'
    args.dataset = data_name
    saved_data = load_best_params(args.dataset)
    if saved_data:
        saved_params = saved_data['best_params']
        args.lr = saved_params['lr']
        args.num_hops = saved_params['num_hops']
        args.hid_dim = saved_params['hid_dim']
        args.weight_decay = saved_params['weight_decay']
        args.dropout = saved_params['dropout']
        args.input_droprate = saved_params['input_droprate']
        args.dropnode_rate = saved_params['dropnode_rate']
        args.kl = saved_params['kl']
        args.dis = saved_params['dis']
    dataset = get_dataset(args.dataset, 12345)
    x_list = MsgPropagation(dataset.x, dataset.adj, args)
    dataset.X_list = x_list
    args.input_dim = dataset.num_node_features
    args.num_class = dataset.num_classes
    args.val = True
    val_accs, loss_meter_avg, best_model, best_acc = train(dataset, args)
    args.val = False
    test_loss, acc, evidence, evidence_a, u_a, target = test(dataset, best_model, args)
    return acc


if __name__ == '__main__':
    datasets = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'Actor', 'chameleon', 'squirrel']

    for data in datasets:
        accs = []
        for seed in range(5):
            set_seeds(seed + 1)
            acc = main(data)
            accs.append(acc)
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        print('Dataset: {:} Avg acc :{:.2f} ± {:.2f}'.format(data, avg_acc, std_acc))