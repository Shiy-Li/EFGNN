from utils import random_drop
from model import EFGNN
import torch
import copy
import torch.optim as optim


def train(dataset, args):
    losses = []
    test_accs = []
    best_acc = 0
    patience_t = 0
    model = EFGNN(args)
    if torch.cuda.is_available():
        model.cuda()
        dataset.y = dataset.y.cuda()
    X_list = dataset.X_list
    target = dataset.y
    mask = dataset.train_mask
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        data_list = []
        for k in range(args.num_hops + 1):
            data_list.append(random_drop(X_list[k], args.dropnode_rate, training=True))
        optimizer.zero_grad()
        evidence, evidence_a, u_a, loss = model(data_list, target, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        losses.append(total_loss)
        if args.val:
            test_loss, test_acc, evidence, evidence_a, u_a, _ = test(dataset, model, args)
            test_accs.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model)
                patience_t = 0
            else:
                patience_t += 1
        if patience_t >= args.patience_period:
            return test_accs, losses, best_model, best_acc
    return test_accs, losses, best_model, best_acc


def test(dataset, test_model, args):
    test_model.eval()
    correct_num, data_num = 0, 0
    X_list = dataset.X_list
    target = dataset.y
    if torch.cuda.is_available():
        target = target.cuda()
    data_list = []
    for k in range(args.num_hops + 1):
        data_list.append(random_drop(X_list[k], args.dropnode_rate, training=False))
    mask = dataset.val_mask if args.val else dataset.test_mask
    data_num += target[mask].size(0)
    with torch.no_grad():
        evidence, evidence_a, u_a, loss = test_model(data_list, target, mask)
        _, predicted = torch.max(evidence_a.data, 1)
    correct_num += (predicted[mask] == target[mask]).sum().item()
    return loss.item(), correct_num / data_num, evidence, evidence_a, u_a, target
