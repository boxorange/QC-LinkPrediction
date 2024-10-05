import os
import csv
import openpyxl  # Used for writing Excel files
import json
import torch
import numpy as np
import pickle as pkl
from collections import defaultdict
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score


train_data_path = os.path.expanduser('~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_up_to_2021.tsv')
valid_data_path = os.path.expanduser('~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_between_2022_and_2022.tsv')
test_data_path = os.path.expanduser('~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_between_2023_and_2024.tsv')

# to convert indices to strings and save them in a file. 
keyword_file = os.path.expanduser("~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_keywords_2024.txt")

# when evaluating isolated nodes, 
only_isolated_node_edges = False

# when saving model predictions by top models in a file,
save_preds = True
cls_idx = 1 # select all edges (-1), positive edges (1), or negative edges (0) for the predictions.
corpus_file_path = os.path.expanduser("~/QC-LinkPrediction/data/quant-ph-06-15-2024.json")

start_year = 2023 
end_year = 2024

if save_preds:
    model_test_edge_result_path_dict = {
        'MLP': [
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/mlp_model/semnet/Mixtral-8x7B-Instruct-v0.1_td_embedding_edges_20240914_170122.pkl'),
        ],
        'GCN': [
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GCN/semnet/Meta-Llama-3-70B_td_embedding_edges_20240718_195138.pkl'),
        ],
        'NCN': [
            os.path.expanduser('~/QC-LinkPrediction/results/NCN/cn1/semnet/gemini_td_embedding_edges_20240716_231103.pkl'),
        ],
    }
    
else:
    model_test_edge_result_path_dict = {
        'MLP': [
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/mlp_model/semnet/deepwalk_gnn_feature_edges_20240717_020810.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/mlp_model/semnet/line_gnn_feature_edges_20240717_060856.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/mlp_model/semnet/n2v_gnn_feature_edges_20240915_175225.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/mlp_model/semnet/gemini_gnn_feature_edges_20240717_054222.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/mlp_model/semnet/Meta-Llama-3-70B_gnn_feature_edges_20240717_022243.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/mlp_model/semnet/Mixtral-8x7B-Instruct-v0.1_gnn_feature_edges_20240717_021557.pkl'),
        ],
        'GCN': [
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GCN/semnet/deepwalk_gnn_feature_edges_20240718_173619.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GCN/semnet/line_gnn_feature_edges_20240718_150844.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GCN/semnet/n2v_gnn_feature_edges_20240718_175328.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GCN/semnet/gemini_gnn_feature_edges_20240718_151409.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GCN/semnet/Meta-Llama-3-70B_gnn_feature_edges_20240718_161724.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GCN/semnet/Mixtral-8x7B-Instruct-v0.1_gnn_feature_edges_20240915_015711.pkl'),
        ],
        'SAGE': [
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/SAGE/semnet/deepwalk_gnn_feature_edges_20240723_154000.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/SAGE/semnet/line_gnn_feature_edges_20240915_191049.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/SAGE/semnet/n2v_gnn_feature_edges_20240723_155125.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/SAGE/semnet/gemini_gnn_feature_edges_20240723_154118.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/SAGE/semnet/Meta-Llama-3-70B_gnn_feature_edges_20240723_163803.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/SAGE/semnet/Mixtral-8x7B-Instruct-v0.1_gnn_feature_edges_20240723_172012.pkl'),	
        ],
        'GAE': [
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GAE/semnet/deepwalk_gnn_feature_edges_20240723_200540.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GAE/semnet/line_gnn_feature_edges_20240723_201248.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GAE/semnet/n2v_gnn_feature_edges_20240723_200538.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GAE/semnet/gemini_gnn_feature_edges_20240723_200529.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GAE/semnet/Meta-Llama-3-70B_gnn_feature_edges_20240717_140337.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/GAE/semnet/Mixtral-8x7B-Instruct-v0.1_gnn_feature_edges_20240717_140344.pkl'),
        ],
        'NCN': [
            os.path.expanduser('~/QC-LinkPrediction/results/NCN/cn1/semnet/deepwalk_embedding_edges_20240716_212553.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/NCN/cn1/semnet/line_embedding_edges_20240717_001421.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/NCN/cn1/semnet/n2v_embedding_edges_20240716_203415.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/NCN/cn1/semnet/gemini_embedding_edges_20240716_201442.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/NCN/cn1/semnet/Meta-Llama-3-70B_embedding_edges_20240716_212947.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/NCN/cn1/semnet/Mixtral-8x7B-Instruct-v0.1_embedding_edges_20240716_220117.pkl'),
        ],
        'BUDDY': [
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/BUDDY/semnet/deepwalk_gnn_feature_edges_20240724_231547.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/BUDDY/semnet/line_gnn_feature_edges_20240724_162844.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/BUDDY/semnet/n2v_gnn_feature_edges_20240724_161803.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/BUDDY/semnet/gemini_gnn_feature_edges_20240724_161251.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/BUDDY/semnet/Meta-Llama-3-70B_gnn_feature_edges_20240724_191751.pkl'),
            os.path.expanduser('~/QC-LinkPrediction/results/GNN/BUDDY/semnet/Mixtral-8x7B-Instruct-v0.1_gnn_feature_edges_20240724_192728.pkl'),
        ],
    }

train_node_set = set()
with open(train_data_path) as fin:
    for line in fin.readlines():
        n1, n2 = line.split('\t')
        n1 = n1.strip()
        n2 = n2.strip()
        train_node_set.update([n1, n2])

valid_node_set = set()
with open(valid_data_path) as fin:
    for line in fin.readlines():
        n1, n2 = line.split('\t')
        n1 = n1.strip()
        n2 = n2.strip()
        valid_node_set.update([n1, n2])

test_node_set = set()
with open(test_data_path) as fin:
    for line in fin.readlines():
        n1, n2 = line.split('\t')
        n1 = n1.strip()
        n2 = n2.strip()
        test_node_set.update([n1, n2])
        
# find isolated nodes in the training data.
isolated_nodes = sorted(list(test_node_set - train_node_set))
# isolated_nodes = sorted(list(test_node_set - train_node_set - valid_node_set))
isolated_nodes = torch.tensor([int(node) for node in isolated_nodes])

# Convert the list to a tensor
isolated_nodes = torch.tensor([int(node) for node in isolated_nodes])

num_of_nodes = len(isolated_nodes) if only_isolated_node_edges else len(train_node_set | test_node_set)
# num_of_nodes = len(isolated_nodes) if only_isolated_node_edges else len(train_node_set | valid_node_set | test_node_set)

# [START] - save model predictions in a file.
if save_preds:
    # Initialize a dictionary to store sets of correctly classified samples for each model
    model_correctly_classified_samples = {}
# [END] - save model predictions in a file. 

for model_name, result_file_list in model_test_edge_result_path_dict.items():
    for result_file in result_file_list:
        with open(result_file, 'rb') as fin:
            model_test_edge_result = pkl.load(fin)
        
        auc_scores = []
        ap_scores = []
        
        # [START] - save model predictions in a file. 
        if save_preds:
            # Initialize a set to store the common correctly classified samples across all epochs for this model
            model_samples_set = None
        # [END] - save model predictions in a file. 
    
        for epoch, run_with_best_valid in enumerate(model_test_edge_result, 1):

            test_edges = run_with_best_valid[2] # test dataset.

            pos_test_pred = test_edges[0]
            pos_test_edges = test_edges[1]
            neg_test_pred = test_edges[2]
            neg_test_edges = test_edges[3]

            X_test = torch.cat([pos_test_edges, neg_test_edges]).cpu()
            y_pred = torch.cat([pos_test_pred, neg_test_pred]).cpu()
            y_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), torch.zeros(neg_test_pred.size(0), dtype=int)]).cpu()
            
            if only_isolated_node_edges:
                # Create a mask to identify rows where any value matches isolated_nodes
                mask = (X_test[:, 0].unsqueeze(1) == isolated_nodes).any(dim=1) | (X_test[:, 1].unsqueeze(1) == isolated_nodes).any(dim=1)
                
                # Remove the rows where the mask is True
                X_test = X_test[mask]
                y_pred = y_pred[mask]
                y_true = y_true[mask]

            auc = roc_auc_score(y_true, y_pred) * 100
            ap = average_precision_score(y_true, y_pred) * 100
            
            auc_scores.append(auc)
            ap_scores.append(ap)
            
            # [START] - save model predictions in a file. 
            if save_preds:
                # find correctly predicted samples based on the optimal threshold.
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)

                # maximize TPR - FPR
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]

                # Classify samples based on the optimal threshold (Youden's J)
                y_pred_optimal = (y_pred >= optimal_threshold).int()

                # Find correctly classified samples
                if cls_idx == -1: # cls_idx = -1 (for all samples)
                    correctly_classified_indices = np.where(y_pred_optimal == y_true)[0]
                else:
                    indices = np.where(y_true == cls_idx)[0]  # cls_idx = 1 (for positive samples), cls_idx = 0 (for negative samples)
                    correctly_classified_indices = indices[np.where(y_pred_optimal[indices] == y_true[indices])[0]]

                # Extract correctly classified positive samples from X_test
                correctly_classified_samples = X_test[correctly_classified_indices]

                # Convert each sample pair to a frozenset (to ignore order of elements in pairs)
                correctly_classified_samples_set = {frozenset(sample.tolist()) for sample in correctly_classified_samples}

                # Initialize or update the set for this model with the intersection across epochs
                if model_samples_set is None:
                    model_samples_set = correctly_classified_samples_set
                else:
                    model_samples_set = model_samples_set.intersection(correctly_classified_samples_set)
                
            # [END] - save model predictions in a file. 

        # [START] - save model predictions in a file. 
        if save_preds:
            # Store the set of common correctly classified samples for this model
            model_correctly_classified_samples[model_name] = model_samples_set
        # [END] - save model predictions in a file. 

        mean_auc = round(np.mean(auc_scores), 2)
        std_auc = round(np.std(auc_scores), 2)
        mean_ap = round(np.mean(ap_scores), 2)
        std_ap = round(np.std(ap_scores), 2)
        
        filename = result_file.rsplit('/', 1)[1]
        
        print('****************************************************')
        print(f'>> Number of Nodes: {num_of_nodes}, Number of Edges: {len(X_test)}')
        print(f">> ({model_name}) Result file: {filename}")
        print(f">> ({model_name}) AUC: {mean_auc:.2f} ± {std_auc:.2f}, AP: {mean_ap:.2f} ± {std_ap:.2f}")
        print('===================================================\n')

# [START] - save model predictions in a file. 
if save_preds:
    # Find the common correctly classified samples across all models
    common_correctly_classified_samples = set.intersection(*model_correctly_classified_samples.values())

    # convert indices to strings.
    # Step 1: Load the keywords from keyword.txt into a dictionary
    keyword_dict = {}
    with open(keyword_file, 'r') as f:
        for line in f:
            # Split each line by tab to get the index and the corresponding keyword
            idx, keyword = line.strip().split('\t')
            keyword_dict[int(idx)] = keyword

    # Step 2: Convert the indices in common_correctly_classified_samples to keywords
    keyword_pairs = []
    for sample in common_correctly_classified_samples:
        # sample is a frozenset, so iterate over it and map each index to its keyword
        keyword_pair = [keyword_dict[index] for index in sample]
        keyword_pair_sorted = sorted(keyword_pair)  # Sort the keyword list to maintain a consistent order
        
        assert len(keyword_pair_sorted) == 2
        
        keyword_pairs.append((keyword_pair_sorted[0], keyword_pair_sorted[1])) # Two keywords

    # Step 3: Sort the keyword pairs alphabetically based on the first keyword, then second
    keyword_pairs_sorted = sorted(keyword_pairs, key=lambda x: (x[0], x[1]))
    

    # when saving only key pairs,
    '''
    # Step 4: Save the sorted keyword pairs to a TSV file with a number in the first column
    if cls_idx == -1:
        prefix = 'all'
    elif cls_idx == 1:
        prefix = 'pos'
    elif cls_idx == 0:
        prefix = 'neg'
        
    outfile_name = f'common_correctly_classified_{prefix}_samples.tsv'
    
    with open(outfile_name, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        # Add header row (optional)
        writer.writerow(['Number', 'Keyword 1', 'Keyword 2'])
        
        # Write each keyword pair with a sequential number
        for idx, (keyword1, keyword2) in enumerate(keyword_pairs_sorted, start=1):
            writer.writerow([idx, keyword1, keyword2])

    print(f"Keyword pairs have been saved in '{outfile_name}' in alphabetical order.")
    '''
    
    # when saving not only key pairs but other information, 
    
    # read the corpus.
    corpus = json.load(open(corpus_file_path)) 
    
    # Initialize a list to store the results
    results = []

    # Iterate through the corpus year by year
    for year, docs in corpus.items():
        print('>> year:', year, '/ number of docs:', len(docs))

        # Only process documents within the specified date range
        if int(year) >= start_year and int(year) <= end_year:
            for doc in docs:
                article_id = doc.get('article_id', '')
                link = doc.get('link', '')
                journal_ref = doc.get('journal_ref', '')
                arXiv_categories = '; '.join(doc.get('arXiv_categories', ''))
                title = doc.get('title', '')
                abstract = doc.get('abstract', '')
                document_text = title + ' ' + abstract
                document_text = document_text.lower()

                # Check each keyword pair for occurrence in the title or abstract
                for keyword1, keyword2 in keyword_pairs_sorted:
                    if keyword1 in document_text and keyword2 in document_text:
                        # If both keywords appear, append the relevant data to the results list
                        results.append([article_id, link, keyword1, keyword2, title, abstract, journal_ref, arXiv_categories])
    
    # Sort the results by article_id
    results.sort(key=lambda x: x[0])
    
    # Step 4: Save the sorted keyword pairs to an Excel file.
    if cls_idx == -1:
        prefix = 'all'
    elif cls_idx == 1:
        prefix = 'pos'
    elif cls_idx == 0:
        prefix = 'neg'
        
    outfile_name = f'common_correctly_classified_{prefix}_samples_with_additional_info.xlsx'
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Keyword Pairs"

    # Write the header row
    ws.append(['Article ID', 'Link', 'Keyword 1', 'Keyword 2', 'Title', 'Abstract', 'Journal Ref', 'arXiv Categories'])

    # Write the results rows
    for row in results:
        ws.append(row)

    # Save the Excel file
    wb.save(outfile_name)
    print(f'Excel file saved as {outfile_name}')
# [END] - save model predictions in a file. 
