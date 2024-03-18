import torch
import argparse

from sklearn.model_selection import StratifiedKFold 

def get_kfold_idx_split(dataset, num_fold=10, random_state=0):
    '''
    StratifiedKFold function
    '''
    skf = StratifiedKFold(num_fold, shuffle=True, random_state=random_state)

    train_indices, augvalid_indices, test_indices = [], [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).long())
    valid_indices = [test_indices[i - 1] for i in range(num_fold)]

    for i in range(num_fold):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[valid_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
        
        augvalid_mask = torch.ones(len(dataset), dtype=torch.uint8)
        augvalid_mask[test_indices[i]] = 0
        augvalid_indices.append(augvalid_mask.nonzero(as_tuple=False).view(-1))

    return {"train": train_indices, "valid": valid_indices, "augvalid": augvalid_indices, "test": test_indices}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value')

