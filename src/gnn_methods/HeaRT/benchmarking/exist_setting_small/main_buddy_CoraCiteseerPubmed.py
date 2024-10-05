import argparse
import time
import warnings
from math import inf
import sys

import os
# [GP] - reset the path. 06/02/2024
# sys.path.append("..")
sys.path.append(os.path.expanduser("~/QC-LinkPrediction/src/gnn_methods/HeaRT/benchmarking"))


from utils import *
import numpy as np
import torch
from ogb.linkproppred import Evaluator
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_networkx, to_undirected
from baseline_models.BUDDY.data import get_loaders
from baseline_models.BUDDY.utils import select_embedding, select_model, get_num_samples, get_loss, get_split_samples, str2bool
from torch.utils.data import DataLoader
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc, get_threshold # [GP] - get an optimal threshold in roc curve for result analysis. 07/15/2024

# [GP] - negative sampling from non-positive edges. 07/06/2024
import shutil
from quantum import Quantum
from torch_geometric.utils import negative_sampling

# [GP] - Measure the processing time.
from datetime import timedelta, datetime

# [GP] - save edges for result analysis. 07/15/2024
import pickle


log_print = get_logger('testrun', 'log', get_config_dir())


def get_data(args):
    data_name = args.data_name
    data_dir = args.data_dir

    data = read_data(data_name, data_dir)

    # [GP] - added SEMNET data load. 07/12/2024
    if data_name == 'semnet':
        
        # [GP] - passed train, valid, test, and embed data paths. 07/12/2024
        raw_data_paths = {
            'processed_dir': args.processed_dir,
            'train_data_path': args.train_data_path,
            'valid_data_path': args.valid_data_path,
            'test_data_path': args.test_data_path,
            'embed_path': args.embed_tsv_path,
        }

        raw_data_paths = {k: os.path.expanduser(v) for k, v in raw_data_paths.items()}
        
        
        ## TODO: find a better way or function.
        import torch_geometric.transforms as T

        def get_transform_semnet():
            transform = T.Compose([
                T.RandomLinkSplit(num_val=0, 
                                  num_test=0, 
                                  is_undirected=True,
                                  split_labels=False, 
                                  add_negative_train_samples=True),
                ])
                
            return transform
            
        # delete an old processed files.
        processed_dir_path = raw_data_paths['processed_dir']
        if os.path.exists(processed_dir_path):
            shutil.rmtree(processed_dir_path)
            print(f">> Directory '{processed_dir_path}' and all its contents have been deleted.")
        else:
            print(f">> Directory '{processed_dir_path}' does not exist.")
            
        train_data = Quantum(root="dataset", name=data_name, raw_data_paths=raw_data_paths, split='train', transform=get_transform_semnet())
        val_data = Quantum(root="dataset", name=data_name, raw_data_paths=raw_data_paths, split='valid', transform=get_transform_semnet())
        test_data = Quantum(root="dataset", name=data_name, raw_data_paths=raw_data_paths, split='test', transform=get_transform_semnet())
        
        train_data = train_data[0][0]
        val_data = val_data[0][0]
        test_data = test_data[0][0]
        
        
        ## TODO: clean this later.
        class dummpy_dataset:
            def __init__(self):
                self.root = ''
                self.num_features = 0
                self.num_nodes = 0

        dataset = dummpy_dataset()
        dataset.root = os.path.expanduser(args.data_dir)
        dataset.num_features = train_data.x.size(1)
        dataset.num_nodes = train_data.x.size(0)

    else:
        dataset = Planetoid('/home/ac.gpark/QC-LinkPrediction/dataset/', data_name)
    
        transform = RandomLinkSplit(is_undirected=True, num_val=0.05, num_test=0.1, add_negative_train_samples=True)
        train_data, val_data, test_data = transform(dataset.data)

    train_pos = data['train']['edge'].t()
    train_pos_re = to_undirected(train_pos)

    train_data.edge_index = train_pos_re
    val_data.edge_index = train_pos_re
    test_data.edge_index = train_pos_re
    train_data.edge_label_index[:, train_data.edge_label==1] = train_pos

    valid_pos = data['valid']['edge'].t()
    valid_neg = data['valid']['edge_neg'].t()
    val_data.edge_label_index[:, val_data.edge_label==1] = valid_pos
    val_data.edge_label_index[:, val_data.edge_label==0] = valid_neg

    
    test_pos = data['test']['edge'].t()
    test_neg = data['test']['edge_neg'].t()
    test_data.edge_label_index[:, test_data.edge_label==1] = test_pos
    test_data.edge_label_index[:, test_data.edge_label==0] = test_neg

    directed = False

    splits = {'train': train_data, 'valid': val_data, 'test': test_data}
    return dataset, splits, directed


# [GP] - added data dir path. 07/12/2024
def read_data(data_name, data_dir):
   
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
   
    ##############
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []
    node_set = set()
    dir_path = get_root_dir()
    
    for split in ['train', 'test', 'valid']:
        
        if data_name == 'semnet':
            path = os.path.join(data_dir, '{}_pos.txt'.format(split))
        else:
            path = '~/QC-LinkPrediction/data/HeaRT' + '/{}/{}_pos.txt'.format(data_name, split)
            path = os.path.expanduser(path)
            
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                

            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    print('the number of nodes in ' + data_name + ' is: ', num_nodes)

    for split in ['test', 'valid']:
        
        if data_name == 'semnet':
            path = os.path.join(data_dir, '{}_neg.txt'.format(split))
        else:
            path = '~/QC-LinkPrediction/data/HeaRT' + '/{}/{}_pos.txt'.format(data_name, split)
            path = os.path.expanduser(path)
        
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            # if sub == obj:
            #     continue
            
            if split == 'valid':  valid_neg.append((sub, obj))
               
            if split == 'test': test_neg.append((sub, obj))

    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)

    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]

    split_edge['train']['edge'] = train_pos_tensor
    # data['train_val'] = train_val

    split_edge['valid']['edge']= valid_pos
    split_edge['valid']['edge_neg'] = valid_neg
    split_edge['test']['edge']  = test_pos
    split_edge['test']['edge_neg']  = test_neg

    return split_edge


def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    
    # [GP] - only measure AUC and AP for now. this saves the processing time. 07/06/2024
    '''
    k_list = [1, 3, 10, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in [1, 3, 10, 100]:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])


    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1) )
    
    # result_mrr = {}
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    # for K in [1,3,10, 100]:
    #     result[f'mrr_hit{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])
    '''
   
    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])

    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])

    # [GP] - get an optimal threshold in roc curve for result analysis. 07/15/2024
    result_thr_train = get_threshold(train_pred, train_true)
    result_thr_val = get_threshold(val_pred, val_true)
    result_thr_test = get_threshold(test_pred, test_true)

    result['Threshold'] = (result_thr_train, result_thr_val, result_thr_test)

    return result


def train_elph(model, optimizer, train_loader, args, device):
    """
    train a GNN that calculates hashes using message passing
    @param model:
    @param optimizer:
    @param train_loader:
    @param args:
    @param device:
    @return:
    """
   
    t0 = time.time()
    model.train()
    total_loss = 0
    data = train_loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    # sampling
    train_samples = get_num_samples(args.train_samples, len(labels))
    sample_indices = torch.randperm(len(labels))[:train_samples]
    links = links[sample_indices]
    labels = labels[sample_indices]

   
    batch_processing_times = []
    loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
    for batch_count, indices in enumerate((loader)):
        # do node level things
        if model.node_embedding is not None:
            if args.propagate_embeddings:
                emb = model.propagate_embeddings_func(data.edge_index.to(device))
            else:
                emb = model.node_embedding.weight
        else:
            emb = None
        # get node features
        node_features, hashes, cards = model(data.x.to(device), data.edge_index.to(device))
        curr_links = links[indices].to(device)
        batch_node_features = None if node_features is None else node_features[curr_links]
        batch_emb = None if emb is None else emb[curr_links].to(device)
        # hydrate link features
        if args.use_struct_feature:
            subgraph_features = model.elph_hashes.get_subgraph_features(curr_links, hashes, cards).to(device)
        else:  # todo fix this
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        start_time = time.time()
        optimizer.zero_grad()
        logits = model.predictor(subgraph_features, batch_node_features, batch_emb)
        loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).to(device))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
        batch_processing_times.append(time.time() - start_time)
   

    return total_loss / len(train_loader.dataset)



def train(model, optimizer, train_loader, args, device, emb=None):
    # print('starting training')
    t0 = time.time()
    model.train()
    total_loss = 0
    data = train_loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    # sampling
    train_samples = get_num_samples(args.train_samples, len(labels))
    sample_indices = torch.randperm(len(labels))[:train_samples]
    links = links[sample_indices]
    labels = labels[sample_indices]

    
    batch_processing_times = []
    loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
    for batch_count, indices in enumerate(loader):
        # do node level things
        if model.node_embedding is not None:
            if args.propagate_embeddings:
                emb = model.propagate_embeddings_func(data.edge_index.to(device))
            else:
                emb = model.node_embedding.weight
        else:
            emb = None
        curr_links = links[indices]
        batch_emb = None if emb is None else emb[curr_links].to(device)

        if args.use_struct_feature:
           
            sf_indices = sample_indices[indices]  # need the original link indices as these correspond to sf
            subgraph_features = data.subgraph_features[sf_indices].to(device)
            
               
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        node_features = data.x[curr_links].to(device)
        degrees = data.degrees[curr_links].to(device)
        if args.use_RA:
            ra_indices = sample_indices[indices]
            RA = data.RA[ra_indices].to(device)
        else:
            RA = None
        start_time = time.time()
        optimizer.zero_grad()
        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).to(device))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
        # batch_processing_times.append(time.time() - start_time)

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_edge(model, loader, device, args, split=None):

    model.eval()
    n_samples = get_split_samples(split, args, len(loader.dataset))
    t0 = time.time()
    preds = []
    
    # [GP] - return edges for result analysis. 07/15/2024
    all_links = []
    
    data = loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    loader = DataLoader(range(len(links)), args.eval_batch_size,
                        shuffle=False)  # eval batch size should be the largest that fits on GPU
    if model.node_embedding is not None:
        if args.propagate_embeddings:
            emb = model.propagate_embeddings_func(data.edge_index.to(device))
        else:
            emb = model.node_embedding.weight
    else:
        emb = None
    for batch_count, indices in enumerate(loader):
        curr_links = links[indices]
        batch_emb = None if emb is None else emb[curr_links].to(device)
        if args.use_struct_feature:
            subgraph_features = data.subgraph_features[indices].to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        node_features = data.x[curr_links].to(device)
        degrees = data.degrees[curr_links].to(device)
        if args.use_RA:
            RA = data.RA[indices].to(device)
        else:
            RA = None
        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        preds.append(logits.view(-1).cpu())
        
        # [GP] - return edges for result analysis. 07/15/2024
        all_links.append(curr_links)
        
        
        
        # print(curr_links)
        # print(curr_links.shape)
        # print(preds)
        # print(preds[0].shape)
        # input('enter..')
        
        
        
        
        if (batch_count + 1) * args.eval_batch_size > n_samples:
            break
            

    pred = torch.cat(preds)
    labels = labels[:len(pred)]
    pos_pred = pred[labels == 1]
    neg_pred = pred[labels == 0]
    
    # [GP] - return edges for result analysis. 07/15/2024
    all_link = torch.cat(all_links)
    pos_edges = all_link[labels == 1]
    neg_edges = all_link[labels == 0]
    
    
    # print(pos_pred)
    # print(neg_pred)
    # print(pos_pred.shape)
    # print(neg_pred.shape)
    # print(pos_edges)
    # print(neg_edges)
    # print(pos_edges.shape)
    # print(neg_edges.shape)
    # input('enter..')
    
    # [GP] - return edges for result analysis. 07/15/2024
    return pos_pred, neg_pred, pos_edges, neg_edges

@torch.no_grad()
def test_edge_elph(model, loader, device, args, split=None):
    n_samples = get_split_samples(split, args, len(loader.dataset))
    t0 = time.time()
    preds = []
    data = loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    loader = DataLoader(range(len(links)), args.eval_batch_size,
                        shuffle=False)  # eval batch size should be the largest that fits on GPU
    # get node features
    if model.node_embedding is not None:
        if args.propagate_embeddings:
            emb = model.propagate_embeddings_func(data.edge_index.to(device))
        else:
            emb = model.node_embedding.weight
    else:
        emb = None
    node_features, hashes, cards = model(data.x.to(device), data.edge_index.to(device))
    for batch_count, indices in enumerate((loader)):
        curr_links = links[indices].to(device)
        batch_emb = None if emb is None else emb[curr_links].to(device)
        if args.use_struct_feature:
            subgraph_features = model.elph_hashes.get_subgraph_features(curr_links, hashes, cards).to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        batch_node_features = None if node_features is None else node_features[curr_links]
        logits = model.predictor(subgraph_features, batch_node_features, batch_emb)
        preds.append(logits.view(-1).cpu())
        if (batch_count + 1) * args.eval_batch_size > n_samples:
            break

    pred = torch.cat(preds)
    labels = labels[:len(pred)]
    pos_pred = pred[labels == 1]
    neg_pred = pred[labels == 0]
    return pos_pred, neg_pred


def get_test_func(model_str):
    if model_str == 'ELPH':
        return test_edge_elph
    elif model_str == 'BUDDY':
        return test_edge
    
@torch.no_grad()
def test(model, evaluator_hit, evaluator_mrr, train_loader, val_loader, test_loader, args, device):

    test_func = get_test_func(args.model)
    
    # [GP] - return edges for result analysis. 07/15/2024
    pos_train_pred, neg_train_pred, pos_train_edges, neg_train_edges = test_func(model, train_loader, device, args, split='train')
    pos_valid_pred, neg_valid_pred, pos_valid_edges, neg_valid_edges = test_func(model, val_loader, device, args, split='val')
    pos_test_pred, neg_test_pred, pos_test_edges, neg_test_edges = test_func(model, test_loader, device, args, split='test')

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred), torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)
    
    # [GP] - 06/20/2024
    # print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    print(f'>> train: {pos_train_pred.size()}, valid_pos: {pos_valid_pred.size()}, valid_neg: {neg_valid_pred.size()}, test_pos: {pos_test_pred.size()}, test_neg: {neg_test_pred.size()}')
        
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]
    
    # [GP] - return edges and thresholds for result analysis. 07/15/2024
    result['test_edges'] = ((pos_train_pred, pos_train_edges, neg_train_pred, neg_train_edges), 
        (pos_valid_pred, pos_valid_edges, neg_valid_pred, neg_valid_edges), 
        (pos_test_pred, pos_test_edges, neg_test_pred, neg_test_edges))
        
    return result, score_emb



def main():
    parser = argparse.ArgumentParser(description='homo')
    
    parser.add_argument('--data_name', type=str, default='cora')
    
    ## gnn setting
    parser.add_argument('--hidden_channels', type=int, default=256)
    
    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt', dest='kill_cnt', default=10, type=int, help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', default='AUC', help='AUC, AP, MRR') # [GP] - changed the default metric to AUC. 07/06/2024
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)

    ##
    parser.add_argument('--model', type=str, default='BUDDY')
    parser.add_argument('--max_hash_hops', type=int, default=2, help='the maximum number of hops to hash')
    parser.add_argument('--floor_sf', type=str2bool, default=0,
                        help='the subgraph features represent counts, so should not be negative. If --floor_sf the min is set to 0')
    parser.add_argument('--minhash_num_perm', type=int, default=128, help='the number of minhash perms')
    parser.add_argument('--hll_p', type=int, default=8, help='the hyperloglog p parameter')
    parser.add_argument('--use_zero_one', type=str2bool,
                        help="whether to use the counts of (0,1) and (1,0) neighbors")
    parser.add_argument('--load_features', action='store_true', help='load node features from disk')
    parser.add_argument('--load_hashes', action='store_true', help='load hashes from disk')
    parser.add_argument('--cache_subgraph_features', action='store_true',
                        help='write / read subgraph features from disk')
    parser.add_argument('--use_feature', type=str2bool, default=True,
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--use_RA', type=str2bool, default=False, help='whether to add resource allocation features')
    parser.add_argument('--sign_k', type=int, default=0)
    parser.add_argument('--num_negs', type=int, default=1, help='number of negatives for each positive')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_node_embedding', action='store_true',
                        help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                        help="load pretrained node embeddings as additional node features")
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--feature_dropout', type=float, default=0.5)
    parser.add_argument('--propagate_embeddings', action='store_true',
                        help='propagate the node embeddings using the GCN diffusion operator')
    parser.add_argument('--add_normed_features', dest='add_normed_features', type=str2bool,
                        help='Adds a set of features that are normalsied by sqrt(d_i*d_j) to calculate cosine sim')
    parser.add_argument('--train_samples', type=float, default=inf, help='the number of training edges or % if < 1')
    parser.add_argument('--use_struct_feature', type=str2bool, default=True,
                        help="whether to use structural graph features as GNN input")
    parser.add_argument('--loss', default='bce', type=str, help='bce or auc')

    parser.add_argument('--dynamic_train', action='store_true',
                        help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--eval_batch_size', type=int, default=1024*64,
                        help='eval batch size should be largest the GPU memory can take - the same is not necessarily true at training time')
    
    parser.add_argument('--no_sf_elph', action='store_true',
                        help='use the structural feature in elph or not')
    parser.add_argument('--feature_prop', type=str, default='gcn',
                        help='how to propagate ELPH node features. Values are gcn, residual (resGCN) or cat (jumping knowledge networks)')
    
    # [GP] - added additional args. 06/14/2024
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--processed_dir', type=str, default='')
    parser.add_argument('--train_data_path', type=str, default='')
    parser.add_argument('--valid_data_path', type=str, default='')
    parser.add_argument('--test_data_path', type=str, default='')
    
    ## [GP] - TODO: combine these two later. 
    parser.add_argument('--embed_path', type=str, default='gnn_feature')
    parser.add_argument('--embed_tsv_path', type=str, default='gnn_feature')
    
    
    parser.add_argument('--n2v_embed_path', type=str, default='', help='used for node2vec')
    parser.add_argument('--save_log', action='store_true', default=False, help='save results log to file')
    
    args = parser.parse_args()
    
    print(args)
    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # dataset = Planetoid('.', 'cora')

    dataset, splits, directed = get_data(args)
    train_loader, train_eval_loader, val_loader, test_loader = get_loaders(args, dataset, splits, directed)

    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
    # [GP] - only measure AUC and AP for now. this saves the processing time. 07/06/2024
    loggers = {
        # 'Hits@1': Logger(args.runs),
        # 'Hits@3': Logger(args.runs),
        # 'Hits@10': Logger(args.runs),
        # 'Hits@100': Logger(args.runs),
        # 'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs),
        'Threshold':Logger(args.runs)
    }
    
    
    # [GP] - Measure the processing time.
    st = time.time()
    
    # [GP] - save results log to file.
    log_data = []
    
    # [GP] - save test edges for result analysis.
    all_test_edges = []
    
    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        log_data.append('#################################          ' + str(run) + '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        
        print('seed: ', seed)
        log_data.append('seed: ' + str(seed))

        init_seed(seed)
        save_path = args.output_dir+'/lr'+str(args.lr) + '_l2'+ str(args.l2)  +'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        # [GP] - added toget the number of nodes from SEMNET. 07/12/2024
        if args.data_name == 'semnet':
            num_nodes = dataset.num_nodes
        else:
            num_nodes = dataset.data.num_nodes

        emb = select_embedding(args, num_nodes, device)
        model, optimizer = select_model(args, dataset, emb, device)
        
        # [GP] - apply a burn-in-period. E.g., random emb produces good results in the beginning, but it never converges. 06/14/2024
        burn_in_period = 20 # epochs
        
        # [GP] - store test edges for result analysis. 07/15/2024
        best_test_edges = None
        
        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            if args.model == 'BUDDY':
                loss = train(model, optimizer, train_loader, args, device)
            elif args.model == 'ELPH':
                loss = train_elph(model, optimizer, train_loader, args, device)

            if epoch % args.eval_steps == 0:

                results_rank, score_emb = test(model, evaluator_hit, evaluator_mrr, train_eval_loader, val_loader, test_loader, args, device)
                
                if epoch > burn_in_period:
                    for key, result in results_rank.items():
                        
                        # [GP] - skip test_edges. 07/15/2024
                        if key == 'test_edges':
                            continue
                            
                        loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        # [GP] - skip test_edges. 07/15/2024
                        if key == 'test_edges':
                            continue
                            
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result

                        log_print.info(
                            f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                                
                        # [GP] - save results log to file.						
                        log_data.append(f'Run: {run + 1:02d}, '
                                        f'Epoch: {epoch:02d}, '
                                        f'Loss: {loss:.4f}, '
                                        f'Train: {100 * train_hits:.2f}%, '
                                        f'Valid: {100 * valid_hits:.2f}%, '
                                        f'Test: {100 * test_hits:.2f}%')
                                        
                    print('---')
                    
                    # [GP] - save results log to file.						
                    log_data.append('---')
                    
                    if epoch > burn_in_period:
                        best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()
                    
                        if best_valid_current > best_valid:
                            best_valid = best_valid_current
                            kill_cnt = 0

                            if args.save:
                                save_emb(score_emb, save_path)
                            
                            # [GP] - save test_edges. 07/15/2024
                            best_test_edges = results_rank['test_edges']
                        
                        else:
                            kill_cnt += 1
                            
                            if kill_cnt > args.kill_cnt: 
                                print("Early Stopping!!")
                                break					
            
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run, log_data=log_data)
        
        # [GP] - save test_edges. 07/15/2024
        all_test_edges.append(best_test_edges)
        
    result_all_run = {}
    for key in loggers.keys():
        print(key)
        
        # [GP] - save results log to file.						
        log_data.append(f'>> Metric: {key}')
        
        best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics(log_data=log_data)

        if key == eval_metric:
            best_metric_valid_str = best_metric
            best_valid_mean_metric = best_valid_mean

        if eval_metric != 'AUC' and key == 'AUC':
            best_auc_valid_str = best_metric
            best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]
        
    
    # print(best_metric_valid_str + ' ' + best_auc_valid_str)
    
    # [GP] - print the input parameters.
    print(args)

    # [GP] - save results log to file.						
    if args.save_log:
        model_name = args.model
        
        output_dir = os.path.join(args.output_dir, model_name)
        output_dir = os.path.join(output_dir, args.data_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        emb_name = args.embed_path.rsplit('/', 1)[1]
        emb_name = emb_name.replace('.tsv', '')
        emb_name = emb_name.replace('.pt', '')
        
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        filename = f"log_{emb_name}_{timestamp_str}.txt"

        with open(os.path.join(output_dir, filename), 'w') as fout:
            for item in log_data:
                fout.write("%s\n" % item)
            
            fout.write("==============\n\n")
            
            dict_ns = vars(args)
            for key, value in dict_ns.items():
                fout.write(f'{key}: {value}\n')

        filename = f"{emb_name}_edges_{timestamp_str}.pkl"
        with open(os.path.join(output_dir, filename), "wb") as f:
            pickle.dump(all_test_edges, f)
            
    # return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":
    main()
