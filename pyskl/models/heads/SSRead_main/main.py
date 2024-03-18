from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

import torch
import torch.optim as optim
import torch.nn.functional as F

from gnn import GNN
from utils import get_kfold_idx_split, str2bool

import argparse
import time
import numpy as np

mcls_criterion = torch.nn.CrossEntropyLoss()

def train(model, device, loader, optimizer_gnn, optimizer_seg):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        pred = model(batch)
        is_labeled = batch.y == batch.y
        
        optimizer_gnn.zero_grad()
        pred_loss = mcls_criterion(pred.to(torch.float32)[is_labeled], batch.y[is_labeled])
        pred_loss.backward(retain_graph=True)
        optimizer_gnn.step()

        optimizer_seg.zero_grad()
        align_loss = model.get_aligncost(batch)
        align_loss.backward(retain_graph=True)
        optimizer_seg.step()

def eval(model, device, loader):
    model.eval()
    
    y_true, y_pred = [], []
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)
            pred = torch.max(pred, dim=1)[1]
                
        y_true.append(batch.y.view(pred.shape))
        y_pred.append(pred)
    
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    correct = y_true == y_pred
    return {'acc': correct.sum().item()/correct.shape[0]}

def main():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    
    # SSRead parameters
    parser.add_argument('--read_op', type=str, default='sum',
                        help='graph readout operation (default: sum)')
    parser.add_argument('--num_position', type=int, default=4,
                        help='number of structural positions (default: 4)')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='smoothing parameter for soft semantic alignment (default: 0.01)')

    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='maximum number of epochs to train (default: 500)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='initial learning rate of the optimizer (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--early_stop', type=int, default=50,
                        help='patience for early stopping criterion (default: 50)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducibility (default: 0)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    
    # Dataset parameters
    parser.add_argument('--datapath', type=str, default="./dataset",
                        help='path to the directory of datasets (default: ./dataset)')
    parser.add_argument('--dataset', type=str, default="NCI1",
                        help='dataset name (default: NCI1)')
    parser.add_argument('--num_fold', type=int, default=10,
                        help='number of fold for cross-validation (default: 10)')

    args = parser.parse_args()
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = TUDataset(root = args.datapath, name = args.dataset)
    num_classes = int(dataset.data.y.max()) + 1
    num_nodefeats = max(1, dataset.num_node_labels)
    num_edgefeats = max(1, dataset.num_edge_labels)

    split_idx = get_kfold_idx_split(dataset, num_fold=args.num_fold, random_state=args.seed)

    valid_list, test_list = [], []
    for fold_idx in range(args.num_fold): 
        train_loader = DataLoader(dataset[split_idx["train"][fold_idx]], batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["augvalid"][fold_idx]], batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
        test_loader = DataLoader(dataset[split_idx["test"][fold_idx]], batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
        
        if args.gnn == 'gcn':
            model = GNN(gnn_type = 'gcn', num_classes = num_classes, num_nodefeats = num_nodefeats, num_edgefeats = num_edgefeats,
                        num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, 
                        read_op = args.read_op, num_position = args.num_position, gamma = args.gamma).to(device)
        elif args.gnn == 'gin':
            model = GNN(gnn_type = 'gin', num_classes = num_classes, num_nodefeats = num_nodefeats, num_edgefeats = num_edgefeats,
                        num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, 
                        read_op = args.read_op, num_position = args.num_position, gamma = args.gamma).to(device)
        else:
            raise ValueError('Invalid GNN type')

        optimizer_gnn = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer_seg = optim.Adam(model.read.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 

        valid_curve, test_curve = [], []
        best_valid_perf, no_improve_cnt = -np.inf, 0
        
        for epoch in range(1, args.max_epochs + 1):
            train(model, device, train_loader, optimizer_gnn, optimizer_seg)

            valid_perf = eval(model, device, valid_loader)
            test_perf = eval(model, device, test_loader)
            print('%3d\t%.6f\t%.6f'%(epoch, valid_perf['acc'], test_perf['acc']))

            valid_curve.append(valid_perf['acc'])
            test_curve.append(test_perf['acc'])

            if no_improve_cnt >= args.early_stop:
                break
            elif valid_perf['acc'] > best_valid_perf:
                best_valid_perf = valid_perf['acc']
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1

        best_val_epoch = np.argmax(np.array(valid_curve))

        print('%2d-fold Valid\t%.6f'%(fold_idx+1, valid_curve[best_val_epoch]))
        print('%2d-fold Test\t%.6f'%(fold_idx+1, test_curve[best_val_epoch]))

        valid_list.append(valid_curve[best_val_epoch])
        test_list.append(test_curve[best_val_epoch])

    print('Valid\t%.6f\tTest\t%.6f'%(sum(valid_list)/len(valid_list), sum(test_list)/len(test_list)))

if __name__ == "__main__":
    main()
