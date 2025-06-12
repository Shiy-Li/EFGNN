import argparse
from utils import get_dataset, MsgPropagation, load_best_params, set_seeds
from train_test import train, test
import numpy as np


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='the train epochs')
    parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--feat-per-layer', type=list, default=[16, 7], help='the feature dimension of per layer')
    parser.add_argument('--hid-dim', type=int, default=32)
    parser.add_argument('--input-dim', type=int, default=16, help='the hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0., help='the dropout')
    parser.add_argument('--dataset', type=str, default='cora', help='the dataset')
    parser.add_argument('--num-class', type=int, default=0, help='the num_class')
    parser.add_argument('--opt', type=str, default='adam', help='the optimizer')
    parser.add_argument('--batch-size', type=int, default=64, help='the batch_size')
    parser.add_argument('--weight-decay', type=float, default=0.02, help='the weight of decay')
    parser.add_argument('--opt-scheduler', type=str, default='none')
    parser.add_argument('--num-hops', type=int, default=16)
    parser.add_argument('--model-type', type=str, default='EFGNN')
    parser.add_argument('--patience-period', type=int, default=300)
    parser.add_argument('--save-model', type=bool, default=False)
    parser.add_argument('--opt-decay-step', type=int, default=[50, 100, 200])
    parser.add_argument('--opt-decay-rate', type=float, default=0.5)
    parser.add_argument('--residual', type=bool, default=False)
    parser.add_argument('--rg', type=float, default=0.9)
    parser.add_argument('--dropnode-rate', type=float, default=0.)
    parser.add_argument('--input-droprate', type=float, default=0.)
    parser.add_argument('--is-val', type=bool, default=True)
    parser.add_argument('--use-bn', type=bool, default=False)
    parser.add_argument('--noise', type=bool, default=True)
    parser.add_argument('--mu', type=float, default=0)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--dp', type=float, default=0.)
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
        # print('Best Params!!!', saved_data)
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
    args.is_val = True
    val_accs, loss_meter_avg, best_model, best_acc = train(dataset, args)
    args.is_val = False
    test_loss, acc, evidence, evidence_a, u_a, target = test(dataset, best_model, args)
    return acc


if __name__ == '__main__':
    datasets = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'Actor', 'chameleon', 'squirrel']
    # datasets = ['Actor', 'chameleon', 'squirrel']

    for data in datasets:
        accs = []
        for seed in range(5):
            set_seeds(seed + 1)
            acc = main(data)
            accs.append(acc)
            # print(f'Dataset: {data}, acc: {acc}')
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        print('Dataset: {:} Avg acc :{:.2f} ± {:.2f}'.format(data, avg_acc, std_acc))