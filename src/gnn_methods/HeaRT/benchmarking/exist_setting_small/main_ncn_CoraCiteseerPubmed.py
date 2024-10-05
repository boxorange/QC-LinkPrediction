import argparse
import numpy as np
import torch
import sys
import os
# [GP] - reset the path. 06/02/2024
# sys.path.append("..")
sys.path.append(os.path.expanduser("~/QC-LinkPrediction/src/gnn_methods/HeaRT/benchmarking"))
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from baseline_models.NCN.model import predictor_dict, convdict, GCN, DropEdge
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch.utils.tensorboard import SummaryWriter
from baseline_models.NCN.util import PermIterator
import time
# from ogbdataset import loaddataset
from typing import Iterable
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from utils import init_seed, Logger, save_emb, get_logger
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc, get_threshold # [GP] - get an optimal threshold in roc curve for result analysis. 07/15/2024

from utils import *

# [GP] 5/6/2024
import shutil
from quantum import Quantum
from torch_geometric.transforms import RandomLinkSplit

# [GP] - Measure the processing time.
from datetime import timedelta, datetime

# [GP] - save edges for result analysis. 07/15/2024
import pickle


log_print = get_logger('testrun', 'log', get_config_dir())


def randomsplit(dataset, data_name):
   
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
   
    ##############
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []
    node_set = set()
    dir_path = get_root_dir()
    
    for split in ['train', 'test', 'valid']:

        path = dir_path+'/dataset' + '/{}/{}_pos.txt'.format(data_name, split)

       
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
    print('the number of nodes in ' + data_name + ' is:', num_nodes)

    for split in ['test', 'valid']:

        path = dir_path+'/dataset' + '/{}/{}_neg.txt'.format(data_name, split)

      
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            # if sub == obj:
            #     continue
            
            if split == 'valid': 
                valid_neg.append((sub, obj))
               
            if split == 'test': 
                test_neg.append((sub, obj))

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

def loaddataset(name, use_valedges_as_input, load=None, raw_data_paths=None):
    
    # [GP] - 05/11/2024
    if name == "semnet":
        # delete an old processed files.
        processed_dir_path = raw_data_paths['processed_dir']
        if os.path.exists(processed_dir_path):
            shutil.rmtree(processed_dir_path)
            print(f">> Directory '{processed_dir_path}' and all its contents have been deleted.")
        else:
            print(f">> Directory '{processed_dir_path}' does not exist.")
            
        train_data = Quantum(root="dataset", name=name, raw_data_paths=raw_data_paths, split='train')
        valid_data = Quantum(root="dataset", name=name, raw_data_paths=raw_data_paths, split='valid')
        test_data = Quantum(root="dataset", name=name, raw_data_paths=raw_data_paths, split='test')

        split_edge = {'train': {}, 'valid': {}, 'test': {}}
        split_edge['train']['edge'] = train_data[0].edge_index.t()
        # transform = RandomLinkSplit(num_val=0.0, num_test=0.0, split_labels=True, is_undirected=True)
        # train, _, _ = transform(train_data[0])
        # split_edge['train']['edge'] = train.pos_edge_label_index.t()
        
        split_edge['valid']['edge'] = valid_data[0].edge_index.t()
        # split_edge['valid']['edge_neg'] = negative_sampling(valid_data[0].edge_index, force_undirected=True).t()
        
        num_nodes = train_data[0].x.size(0)
        
        # Ensure the edge index is undirected
        # This can be done by including both directions of each edge if not already included
        concat_edge_index = torch.cat([train_data[0].edge_index, train_data[0].edge_index.flip(0)], dim=1)
        concat_edge_index = torch.cat([concat_edge_index, valid_data[0].edge_index], dim=1)
        concat_edge_index = torch.cat([concat_edge_index, valid_data[0].edge_index.flip(0)], dim=1)
        val_num_neg_samples = valid_data[0].edge_index.size(1)
        valid_neg_edge_label_index = negative_sampling(
            edge_index=concat_edge_index,
            num_nodes=num_nodes, 
            num_neg_samples=val_num_neg_samples, 
            # force_undirected=True
        )
        split_edge['valid']['edge_neg'] = valid_neg_edge_label_index.t()
        
        split_edge['test']['edge'] = test_data[0].edge_index.t()
        # split_edge['test']['edge_neg'] = negative_sampling(test_data[0].edge_index, force_undirected=True).t()

        concat_edge_index = torch.cat((concat_edge_index, valid_neg_edge_label_index), dim=1)		
        concat_edge_index = torch.cat((concat_edge_index, valid_neg_edge_label_index.flip(0)), dim=1)		
        concat_edge_index = torch.cat((concat_edge_index, test_data[0].edge_index), dim=1)
        concat_edge_index = torch.cat((concat_edge_index, test_data[0].edge_index.flip(0)), dim=1)
        test_num_neg_samples = test_data[0].edge_index.size(1)
        test_neg_edge_label_index = negative_sampling(
            edge_index=concat_edge_index,
            num_nodes=num_nodes, 
            num_neg_samples=test_num_neg_samples, 
            # force_undirected=True
        )
        split_edge['test']['edge_neg'] = test_neg_edge_label_index.t()
        
        data = train_data[0]
        # data = train_data[0].update(test_data[0])
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]

    else:
        dataset = Planetoid(root="dataset", name=name)
        name = name.lower()
        split_edge = randomsplit(dataset, name)
        data = dataset[0]
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
 
    data.edge_weight = None 
    
    
    # if data.edge_weight is None else data.edge_weight.view(-1).to(torch.float)
    # data = T.ToSparseTensor()(data)
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    
    
    # [GP] - 05/11/2024
    if name == "semnet":
        # debug - testing. it doesn't affect the performance as expected. 06/02/2024
        # feature_embeddings = torch.load('/home/ac.gpark/QC-LinkPrediction/data/SEMNET/LPFormer/gemini_td_gnn_feature')
        # feature_embeddings = feature_embeddings['entity_embedding']
        # data.x = feature_embeddings
        pass
        
    else:
        dir_path = get_root_dir()
        feature_embeddings = torch.load(dir_path+'/dataset' + '/{}/{}'.format(name, 'gnn_feature'))
        feature_embeddings = feature_embeddings['entity_embedding']

        data.x = feature_embeddings
        
    data.max_x = -1

    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])


    # Use training + validation edges for inference on test set.
    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    
    return data, split_edge



def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    # [GP] - This was commented out since SEMNET only utilizes AP and AUC evaluations, as MRR evaluations are excessively time-consuming for SEMNET. 05/11/2024
    result = {}

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
    
def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None):
    
    def penalty(posout, negout):
        scale = torch.ones_like(posout[[0]]).requires_grad_()
        loss = -F.logsigmoid(posout*scale).mean()-F.logsigmoid(-negout*scale).mean()
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(torch.square(grad))
    
    if alpha is not None:
        predictor.setalpha(alpha)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    
    for perm in PermIterator(adjmask.device, adjmask.shape[0], batch_size):
        optimizer.zero_grad()
        
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei, sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        
        else:
            adj = data.adj_t
        
        h = model(data.x, adj)
        
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        pos_losss = -F.logsigmoid(pos_outs).mean()

        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        
        loss = neg_losss + pos_losss
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss


@torch.no_grad()
def test(model, predictor, data, split_edge,  evaluator_hit, evaluator_mrr, batch_size, use_valedges_as_input):
    model.eval()
    predictor.eval()

    # pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)
    
    def test_edge(predictor, input_data, h, adj, batch_size):
        test_edges = []
        preds = []
        for perm in PermIterator(input_data.device, input_data.shape[0], batch_size, False):
            edge = input_data[perm].t()
            preds += [predictor(h, adj, edge).squeeze().cpu()]
            test_edges += [input_data[perm]]
            
        pred_all = torch.cat(preds, dim=0)
        test_edges = torch.cat(test_edges, dim=0)

        return pred_all, test_edges
    
    
    # [GP] - return edges for result analysis. 07/15/2024
    pos_valid_pred, pos_valid_edges = test_edge(predictor, pos_valid_edge, h, adj, batch_size)
    neg_valid_pred, neg_valid_edges = test_edge(predictor, neg_valid_edge, h, adj, batch_size)
    
    pos_test_pred, pos_test_edges = test_edge(predictor, pos_test_edge, h, adj, batch_size)
    neg_test_pred, neg_test_edges = test_edge(predictor, neg_test_edge, h, adj, batch_size)
    
    # [GP] - 05/11/2024
    print(f'>> valid_pos: {pos_valid_pred.size()}, valid_neg: {neg_valid_pred.size()}, test_pos: {pos_test_pred.size()}, test_neg: {neg_test_pred.size()}')
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_valid_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    
    score_emb = [pos_valid_pred.cpu(), neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), h.cpu()]
    
    # [GP] - return edges and thresholds for result analysis. 07/15/2024
    result['test_edges'] = ((None, None), 
        (pos_valid_pred, pos_valid_edges, neg_valid_pred, neg_valid_edges), 
        (pos_test_pred, pos_test_edges, neg_test_pred, neg_test_edges))
        
    return result, score_emb


def parseargs():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--mplayers', type=int, default=1)
    parser.add_argument('--nnlayers', type=int, default=3)
    parser.add_argument('--ln', action="store_true")
    parser.add_argument('--lnnn', action="store_true")
    parser.add_argument('--res', action="store_true")
    parser.add_argument('--jk', action="store_true")
    parser.add_argument('--maskinput', action="store_true")
    parser.add_argument('--hiddim', type=int, default=32)
    parser.add_argument('--gnndp', type=float, default=0.3)
    parser.add_argument('--xdp', type=float, default=0.3)
    parser.add_argument('--tdp', type=float, default=0.3)
    parser.add_argument('--gnnedp', type=float, default=0.3)
    parser.add_argument('--predp', type=float, default=0.3)
    parser.add_argument('--preedp', type=float, default=0.3)
    parser.add_argument('--splitsize', type=int, default=-1)
    parser.add_argument('--gnnlr', type=float, default=0.0003)
    parser.add_argument('--prelr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--testbs', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--trndeg', type=int, default=-1)
    parser.add_argument('--tstdeg', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default="collab")
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument('--model', choices=convdict.keys())
    parser.add_argument('--cndeg', type=int, default=-1)
    parser.add_argument('--save_gemb', action="store_true")
    parser.add_argument('--load', type=str)
    parser.add_argument('--cnprob', type=float, default=0)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--increasealpha", action="store_true")
    parser.add_argument("--savex", action="store_true")
    parser.add_argument("--loadx", action="store_true")
    parser.add_argument("--loadmod", action="store_true")
    parser.add_argument("--savemod", action="store_true")

    ###
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--kill_cnt', dest='kill_cnt', default=10, type=int, help='early stopping')
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--eval_steps', type=int, default=5)
    
    # [GP] - added processed_dir, train, valid, test, and embedding data paths. 06/02/2024
    parser.add_argument('--processed_dir', type=str, default='')
    parser.add_argument('--train_data_path', type=str, default='')
    parser.add_argument('--valid_data_path', type=str, default='')
    parser.add_argument('--test_data_path', type=str, default='')
    parser.add_argument('--embed_path', type=str, default='')
    parser.add_argument('--save_log', action='store_true', default=False, help='save results log to file')
    
    args = parser.parse_args()
    return args


def main():
    args = parseargs()

    print(args, flush=True)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

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
    
    
    # [GP] - passed train, valid, test, and embed data paths. 06/02/2024
    raw_data_paths = {
        'processed_dir': args.processed_dir,
        'train_data_path': args.train_data_path,
        'valid_data_path': args.valid_data_path,
        'test_data_path': args.test_data_path,
        'embed_path': args.embed_path,
    }
    
    data, split_edge = loaddataset(args.dataset, args.use_valedges_as_input, args.load, raw_data_paths)

    data = data.to(device)

    predfn = predictor_dict[args.predictor]

    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    
    # [GP] - Measure the processing time.
    st = time.time()
    
    # [GP] - save results log to file.
    log_data = []
    
    # [GP] - save test edges for result analysis.
    all_test_edges = []
    
    for run in range(0, args.runs):
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        
        print('seed:', seed)
        log_data.append('seed: ' + str(seed))
        
        init_seed(seed)

        save_path = args.output_dir+'/lr'+str(args.gnnlr) + '_drop' + str(args.gnndp) + '_l2'+ str(args.l2) + '_numlayer' + str(args.mplayers)+ '_numPredlay' + str(args.nnlayers) +'_dim'+str(args.hiddim) + '_'+ 'best_run_'+str(seed)

        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp, xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
       
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
       
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, {'params': predictor.parameters(), 'lr': args.prelr}], weight_decay=args.l2)
        
        
        # [GP] - apply a burn-in-period. E.g., random emb produces good results in the beginning, but it never converges. 06/14/2024
        burn_in_period = 20 # epochs
        
        # [GP] - store test edges for result analysis. 07/15/2024
        best_test_edges = None
        
        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer, args.batch_size, args.maskinput, [], alpha)
            # print(f"trn time {time.time()-t1:.2f} s", flush=True)
            
            t1 = time.time()
            if epoch % args.eval_steps == 0:
                results, score_emb = test(model, predictor, data, split_edge, evaluator_hit, evaluator_mrr, args.testbs, args.use_valedges_as_input)
                # print(f"test time {time.time()-t1:.2f} s")
                
                for key, result in results.items():
                    
                    # [GP] - skip test_edges. 07/15/2024
                    if key == 'test_edges':
                        continue
                
                    _, valid_hits, test_hits = result
                    
                    if epoch > burn_in_period:	
                        loggers[key].add_result(run, result)
                        
                    print(key)
                    
                    log_print.info(
                        f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Valid: {100 * valid_hits:.2f}%, '
                        f'Test: {100 * test_hits:.2f}%'
                    )
                    
                    # [GP] - save results log to file.						
                    log_data.append(f'Run: {run + 1:02d}, '
                                    f'Epoch: {epoch:02d}, '
                                    f'Loss: {loss:.4f}, '
                                    f'Valid: {100 * valid_hits:.2f}%, '
                                    f'Test: {100 * test_hits:.2f}%')
                
                print('---', flush=True)
                
                # [GP] - save results log to file.						
                log_data.append('---')
                
                if epoch > burn_in_period:
                    best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max().item()

                    if best_valid_current > best_valid:
                        best_valid = best_valid_current
                        kill_cnt = 0
                        if args.save:
                            save_emb(score_emb, save_path)
                        
                        # [GP] - save test_edges. 07/15/2024
                        best_test_edges = results['test_edges']
                        
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
            
        if key == 'AUC':
            best_auc_valid_str = best_metric
            best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]
        
    print(best_metric_valid_str + ' ' + best_auc_valid_str)
    
    # [GP] - print the input parameters.
    print(args, flush=True)
    
    # [GP] - Measure the processing time.
    et = time.time()
    elapsed_time = et - st
    exec_time = timedelta(seconds=elapsed_time)
    exec_time = str(exec_time)
    print('>> Execution time in hh:mm:ss:', exec_time)
    log_data.append('>> Execution time in hh:mm:ss: ' + exec_time)
    
    # [GP] - save results log to file.						
    if args.save_log:
        model_name = args.predictor
        
        output_dir = os.path.join(args.output_dir, model_name)
        output_dir = os.path.join(output_dir, args.dataset)
        
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
  