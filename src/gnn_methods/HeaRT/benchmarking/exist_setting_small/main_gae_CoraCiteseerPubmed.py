import os
import sys
# [GP] - reset the path. 06/02/2024
# sys.path.append("..")
sys.path.append(os.path.expanduser("~/QC-LinkPrediction/src/gnn_methods/HeaRT/benchmarking"))

import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from gnn_model import *
from utils import *
from scoring import mlp_score
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc, get_threshold # [GP] - get an optimal threshold in roc curve for result analysis. 07/15/2024

# [GP] - Measure the processing time.
import time
from datetime import timedelta, datetime

# [GP] - save edges for result analysis. 07/15/2024
import pickle


dir_path  = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())

# [GP] - added data dir, embedding data paths. 06/14/2024
def read_data(data_name, neg_mode, data_dir, embed_path):
    data_name = data_name

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    for split in ['train', 'test', 'valid']:

        if neg_mode == 'equal':
            # path = dir_path+'/dataset' + '/{}/{}_pos.txt'.format(data_name, split)
            path = os.path.join(data_dir, '{}_pos.txt'.format(split))
        
        ## [TODO] - complete this later. 
        elif neg_mode == 'all':
            # path = dir_path+'/dataset' + '/{}/allneg/{}_pos.txt'.format(data_name, split)
            path = os.path.join(data_dir, 'allneg/{}_pos.txt'.format(split))

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

        if neg_mode == 'equal':
            # path = dir_path+'/dataset' + '/{}/{}_neg.txt'.format(data_name, split)
            path = os.path.join(data_dir, '{}_neg.txt'.format(split))
        
        ## [TODO] - complete this later. 
        elif neg_mode == 'all':
            # path = dir_path+'/dataset' + '/{}/allneg/{}_neg.txt'.format(data_name, split)
            path = os.path.join(data_dir, 'allneg/{}_neg.txt'.format(split))

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            # if sub == obj:
            #     continue
            
            if split == 'valid': 
                valid_neg.append((sub, obj))
               
            if split == 'test': 
                test_neg.append((sub, obj))

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))


    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
          

    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)

    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]


    # feature_embeddings = torch.load(dir_path+'/dataset' + '/{}/{}'.format(data_name, 'gnn_feature'))
    # feature_embeddings = torch.load(os.path.join(data_dir, embed_name))
    feature_embeddings = torch.load(embed_path)
    feature_embeddings = feature_embeddings['entity_embedding']

    data = {}
    data['adj'] = adj
    data['train_pos'] = train_pos_tensor
    data['train_val'] = train_val

    data['valid_pos'] = valid_pos
    data['valid_neg'] = valid_neg
    data['test_pos'] = test_pos
    data['test_neg'] = test_neg

    data['x'] = feature_embeddings

    return data


def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    
    # [GP] - only measure AUC and AP for now. this saves the processing time. 07/06/2024
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

        

def train(model, adj, x, optimizer, with_loss_weight):
    model.train()
    # score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0

    optimizer.zero_grad()
    h = model(x, adj)
    inner_prod = torch.sigmoid(torch.mm(h, h.t()))

    # loss = torch.norm((adj.to_dense()-inner_prod), p = 'fro')
    ###############
    if with_loss_weight:
        # print('using loss weight')
        pos_weight = float(adj.size(0) * adj.size(0) - adj.sum()) / adj.sum()
        weight_mask = adj.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(x.device)
        weight_tensor[weight_mask] = pos_weight
    #########################

        loss = F.binary_cross_entropy(inner_prod.view(-1), adj.to_dense().view(-1), weight=weight_tensor)
    else:
        loss = F.binary_cross_entropy(inner_prod.view(-1), adj.to_dense().view(-1))

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   

    optimizer.step()

    return loss


@torch.no_grad()
def test_edge( input_data, h, batch_size):
    
    # [GP] - return edges for result analysis. 07/15/2024
    test_edges = []
    
    # input_data  = input_data.transpose(1, 0)
    preds = []
    for perm  in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()
        src, dst= h[edge[0]], h[edge[1]]
        preds += [torch.sigmoid((src*dst).sum(-1)).cpu()]

        # preds += [ inner_prod[edge[0], edge[1]].cpu()]
        
        # [GP] - return edges for result analysis. 07/15/2024
        test_edges += [input_data[perm]]
        
    pred_all = torch.cat(preds, dim=0)

    # [GP] - return edges for result analysis. 07/15/2024
    test_edges = torch.cat(test_edges, dim=0)

    return pred_all, test_edges


@torch.no_grad()
def test(model, data, x, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    # score_func.eval()

    # adj_t = adj_t.transpose(1,0)
    
    
    h = model(x, data['adj'].to(x.device))

    # inner_prod = torch.sigmoid(torch.mm(h, h.t()))

    
    # [GP] - return edges for result analysis. 07/15/2024
    pos_train_pred, pos_train_edges = test_edge(data['train_val'], h, batch_size)

    neg_valid_pred, neg_valid_edges = test_edge(data['valid_neg'], h, batch_size)
    pos_valid_pred, pos_valid_edges = test_edge(data['valid_pos'], h, batch_size)

    pos_test_pred, pos_test_edges = test_edge(data['test_pos'], h, batch_size)
    neg_test_pred, neg_test_edges = test_edge(data['test_neg'], h, batch_size)

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)

    # [GP] - 06/20/2024
    # print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    print(f'>> train: {pos_train_pred.size()}, valid_pos: {pos_valid_pred.size()}, valid_neg: {neg_valid_pred.size()}, test_pos: {pos_test_pred.size()}, test_neg: {neg_test_pred.size()}')
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    
    # [GP] - return edges and thresholds for result analysis. 07/15/2024
    result['test_edges'] = ((pos_train_pred, pos_train_edges), 
        (pos_valid_pred, pos_valid_edges, neg_valid_pred, neg_valid_edges), 
        (pos_test_pred, pos_test_edges, neg_test_pred, neg_test_edges))
        
    return result


def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ## gnn setting
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)

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
    parser.add_argument('--metric', type=str, default='AUC', help='AUC, AP, MRR') # [GP] - changed the default metric to AUC. 07/06/2024
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    
    # [GP] - added gat_head arg. 07/12/2024
    ###### gat
    parser.add_argument('--gat_head', type=int, default=1)
    
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    #####
    parser.add_argument('--with_loss_weight', default=False, action='store_true')

    # [GP] - added data dir, embedding data paths. 06/14/2024
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--embed_path', type=str, default='gnn_feature')
    parser.add_argument('--save_log', action='store_true', default=False, help='save results log to file')
    
    args = parser.parse_args()
   
   
    # print(args.lr, args.l2, args.dropout)
    print(args.with_loss_weight)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    # [GP] - added data dir, embedding data paths. 06/14/2024
    data_dir = os.path.expanduser(args.data_dir)
    embed_path = args.embed_path
    # data = read_data(args.data_name, args.neg_mode)
    data = read_data(args.data_name, args.neg_mode, data_dir, embed_path)

    input_channel = data['x'].size(1)
    
    # [GP] - added more parameters. 07/12/2024
    node_num = data['x'].size(0)
    
    # model = eval(args.gnn_model)(input_channel, args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout).to(device)
    model = eval(args.gnn_model)(input_channel, args.hidden_channels, args.hidden_channels, args.num_layers, args.dropout, args.gin_mlp_layer, args.gat_head, node_num).to(device)
    
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels, 1, args.num_layers_predictor, args.dropout).to(device)
   
    x = data['x'].to(device)
    train_pos = data['train_pos'].to(x.device)
    adj = data['adj'].to(device)

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
        save_path = args.output_dir+'/best_run_'+str(run)

        model.reset_parameters()
        score_func.reset_parameters()

        optimizer = torch.optim.Adam(
                list(model.parameters()),lr=args.lr, weight_decay=args.l2)
        
        # [GP] - apply a burn-in-period. E.g., random emb produces good results in the beginning, but it never converges. 06/14/2024
        burn_in_period = 20 # epochs
        
        # [GP] - store test edges for result analysis. 07/15/2024
        best_test_edges = None
        
        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, adj, x, optimizer, args.with_loss_weight)
        
            if epoch % args.eval_steps == 0:
                results_rank = test(model, data, x, evaluator_hit, evaluator_mrr, args.batch_size)
                
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
                            save_model(model, save_path, emb=None)
                            
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

        if key == 'AUC':
            best_auc_valid_str = best_metric
            best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]

    # print(best_metric_valid_str +' ' +best_auc_valid_str)
    
    # [GP] - print the input parameters.
    print(args)
    
    # [GP] - Measure the processing time.
    et = time.time()
    elapsed_time = et - st
    exec_time = timedelta(seconds=elapsed_time)
    exec_time = str(exec_time)
    print('>> Execution time in hh:mm:ss:', exec_time)
    log_data.append('>> Execution time in hh:mm:ss: ' + exec_time)
    
    # [GP] - save results log to file.						
    if args.save_log:
        model_name = 'GAE'
            
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
   