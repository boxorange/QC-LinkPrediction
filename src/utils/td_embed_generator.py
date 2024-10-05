import os
import json
import csv
import torch
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from scipy import sparse
from scipy.sparse import linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def find_keyword_cooccurrences(documents, keywords):
    # Initialize a Counter to hold the co-occurrences
    cooccurrences = Counter()
    
    # Iterate over each document
    for doc in documents:
        # Find all the keywords present in the document
        present_keywords = [keyword for keyword in keywords if keyword in doc]
        
        # Get all combinations of two keywords that are present in the document
        for combo in combinations(present_keywords, 2):
            # Update the count for this combination in the Counter
            cooccurrences[combo] += 1
    
    return cooccurrences


def generate_cooccurrence_counter(corpus, final_year, keywords, kw2idx):
    cooccurrence_by_year = {}

    for year, docs in corpus.items():
        if int(year) > final_year:
            continue
        
        documents = [x['title'] + ' ' + x['abstract'] for x in docs]
    
        counter = find_keyword_cooccurrences(documents, keywords.values())

        # build sparse co-occurrence matrix (word-word count matrix).
        row_ind, col_ind, data = zip(*[(kw2idx[i], kw2idx[j], count) for (i, j), count in counter.items()])
        wwcnt_mat = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(keywords), len(keywords)))

        cooccurrence_by_year[year] = {'counter': counter, 'wwcnt_mat': wwcnt_mat}
        
    print('<generate_cooccurrence_counter()> Done')
    
    return cooccurrence_by_year
    
    
def generate_matrices(cooccurrence_by_year, kw2idx):
    """
    ref: https://www.kaggle.com/code/claudecoulombe/word-vectors-from-pmi-matrix/notebook
    
    """
    matrices_by_year = {}
    
    for year, cooccurrence_data in cooccurrence_by_year.items():
        counter = cooccurrence_data['counter']
        wwcnt_mat = cooccurrence_data['wwcnt_mat']
        
        total_count = sum(counter.values())

        # for creating sparse matrices
        row_indxs = []
        col_indxs = []

        # pmi: pointwise mutual information
        pmi_dat_values = []
        # ppmi: positive pointwise mutual information
        ppmi_dat_values = []
        # spmi: smoothed pointwise mutual information
        spmi_dat_values = []
        # sppmi: smoothed positive pointwise mutual information
        sppmi_dat_values = []

        # Sum over words and contexts
        sum_over_words = np.array(wwcnt_mat.sum(axis=0)).flatten()
        sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()

        # Smoothing
        # According to [Levy, Goldberg & Dagan, 2015], the smoothing operation should be done on the context.
        alpha = 0.75
        nca_denom = np.sum(sum_over_contexts**alpha)
        # sum_over_words_alpha = sum_over_words**alpha
        sum_over_contexts_alpha = sum_over_contexts**alpha
        
        ii = 0
        for (kw1, kw2), count in counter.items():
            ii += 1
            if ii % 1000000 == 0:
                print(f'finished {ii/len(counter):.2%} of co-occurrence pairs.')
            kw1_idx = kw2idx[kw1]
            kw2_idx = kw2idx[kw2]
                
            nwc = count
            Pwc = nwc / total_count

            nw = sum_over_contexts[kw1_idx]
            Pw = nw / total_count
            
            nc = sum_over_words[kw2_idx]
            Pc = nc / total_count
            
            pmi = np.log2(Pwc/(Pw*Pc))
            ppmi = max(pmi, 0)
            
            # nca = sum_over_words_alpha[kw2_idx]
            nca = sum_over_contexts_alpha[kw2_idx]
            Pca = nca / nca_denom

            spmi = np.log2(Pwc/(Pw*Pca))
            sppmi = max(spmi, 0)
            
            row_indxs.append(kw1_idx)
            col_indxs.append(kw2_idx)
            pmi_dat_values.append(pmi)
            ppmi_dat_values.append(ppmi)
            spmi_dat_values.append(spmi)
            sppmi_dat_values.append(sppmi)

        pmi_mat = sparse.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)), shape=(len(kw2idx), len(kw2idx)))
        ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)), shape=(len(kw2idx), len(kw2idx)))
        spmi_mat = sparse.csr_matrix((spmi_dat_values, (row_indxs, col_indxs)), shape=(len(kw2idx), len(kw2idx)))
        sppmi_mat = sparse.csr_matrix((sppmi_dat_values, (row_indxs, col_indxs)), shape=(len(kw2idx), len(kw2idx)))

        matrices_by_year[year] = {'pmi_mat': pmi_mat, 'ppmi_mat': ppmi_mat}

    print('<generate_matrices()> Done')
    
    return matrices_by_year


def apply_svd_and_generate_embeds(time_decayed_matrices, embedding_size, n_top_dimensions_to_remove=0):
    svd_vecs = {}
    
    for mat_type, matrix in time_decayed_matrices.items():
        rng = np.random.RandomState(42)
        svd = TruncatedSVD(n_components=embedding_size, random_state=rng)
        word_embeddings = svd.fit_transform(matrix)
        
        if n_top_dimensions_to_remove > 0:
            word_embeddings = word_embeddings[:, n_top_dimensions_to_remove:]				
        
        word_embeddings_norm = normalize(word_embeddings, norm='l2')
        svd_vecs[mat_type] = word_embeddings_norm
        
    print('<apply_svd_and_generate_embeds()> Done')
    
    return svd_vecs


def apply_time_decay(matrices_by_year, decay_rate=0.2):
    """
    generated code by MS Co-Pilot. 05/28/2024
    
    """
    stacked_matrices = {}
    for year, matrices in matrices_by_year.items():
        for mat_type, matrix in matrices.items():
            if mat_type in stacked_matrices:
                stacked_matrices[mat_type].append(matrix)
            else:
                stacked_matrices[mat_type] = [matrix]

    stacked_matrices = {k: np.stack([csr.toarray() for csr in v], axis=0) for k, v in stacked_matrices.items()}
    
    time_decayed_matrices = {}
    
    for mat_type, matrices in stacked_matrices.items():
        # data is a 3D numpy array of shape (t, m, n)
        t, m, n = matrices.shape

        # Create a time decay factor using exponential decay
        decay_factor = np.exp(-decay_rate * np.arange(t)) # positive exponential decay (recent year matrix gets higher score.)

        # Reshape decay_factor to match the data's shape
        decay_factor = decay_factor.reshape(-1, 1, 1)

        # Apply the decay factor to the data
        weighted_data = matrices * decay_factor
        
        # Sum the vectors.
        aggregated_vec = weighted_data.sum(axis=0)
        
        time_decayed_matrices[mat_type] = aggregated_vec

    return time_decayed_matrices
    
    
def concatenate_and_save_embeds(llm_embed_file_path, svd_vecs, tsv_file_path, pt_file_path):
    feature_list = []

    with open(llm_embed_file_path) as fin, open(tsv_file_path, 'w') as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t")
        
        ppmi = svd_vecs['ppmi_mat']

        for row, tc_embed in zip(reader, ppmi):
            feature = row[1:-1]
            feature = [float(x) for x in feature]
            
            feature = 0.8 * torch.tensor(feature) + 0.2 * tc_embed
            feature = feature.tolist()

            feature_list.append(feature)
            writer.writerow(row[:1] + feature + row[-1:])

    feature_dict = {'entity_embedding': torch.tensor(feature_list, dtype=torch.float32)}
    
    if os.path.exists(pt_file_path) == False:
        torch.save(feature_dict, pt_file_path)


def main():
    # quantum physics corpus file from arXiv. 
    corpus_file = os.path.expanduser("~/QC-LinkPrediction/data/quant-ph-06-15-2024.json")
    
    # set the parameters.
    final_year = 2021 # 2019, 2012 # the final year to be used for training.
    decay_rate = 0.2 # time decay rate.
    embedding_size = 768 # 768 (gemini emb size) / 3 = 256
    n_top_dimensions_to_remove = 0 # the number of top dimensions to remove in SVD.
    
    # SEMNET concept list file.
    concept_file = os.path.expanduser("~/QC-LinkPrediction-WIP/data/SEMNET/arxiv_qc_semnet_keywords_2024.txt")
    
    # LLM embedding file list.
    llm_embed_file_path_list = [
        os.path.expanduser("~/QC-LinkPrediction/data/SEMNET/embeds/tsv/gemini_embedding.tsv"),
        os.path.expanduser("~/QC-LinkPrediction/data/SEMNET/embeds/tsv/Meta-Llama-3-70B_embedding.tsv"),
        os.path.expanduser("~/QC-LinkPrediction/data/SEMNET/embeds/tsv/Mixtral-8x7B-Instruct-v0.1_embedding.tsv"),
    ]
    
    td_emb_dir = "~/QC-LinkPrediction-WIP/data/SEMNET/embeds/tsv"

    # read the corpus. 
    corpus = json.load(open(corpus_file)) 
    
    # read keywords.
    keyword_file = open(concept_file)
    keywords = {}
    for x in keyword_file.readlines():
        id, keyword = x.split('\t', 1)
        keywords[int(id.strip())] = keyword.strip()
    
    kw2idx = {value: key for key, value in keywords.items()}

    # generate a co-occurrence counter.
    cooccurrence_by_year = generate_cooccurrence_counter(corpus, final_year, keywords, kw2idx)
    
    # generate matrices.
    matrices_by_year = generate_matrices(cooccurrence_by_year, kw2idx)
    
    # apply time decay.
    time_decayed_matrices = apply_time_decay(matrices_by_year, decay_rate)
    
    # generate embeddings after applying SVD.
    svd_vecs = apply_svd_and_generate_embeds(time_decayed_matrices, embedding_size, n_top_dimensions_to_remove)
    
    for llm_embed_file_path in llm_embed_file_path_list:
        out_dir, out_file_name = llm_embed_file_path.rsplit('/', 1)[1]
        out_file_name = out_file_name.replace('_embedding', '_td_embedding')
        
        tsv_file = out_file_name
        tsv_file_path = os.path.join(out_dir, tsv_file)
        
        pt_file = out_file_name.replace('.tsv', '')
        pt_file_path = os.path.join(out_dir, pt_file)
        
        # concatenate with the LLM embeddings. 
        concatenate_and_save_embeds(llm_embed_file_path, svd_vecs, tsv_file_path, pt_file_path)
        
    # save time decay embeddings.
    ppmi = svd_vecs['ppmi_mat']
    ppmi_list = ppmi.tolist()
    
    output_path = os.path.join(td_emb_dir, 'td_embedding')
    
    with open(output_path + '.tsv', 'w') as fout:
        csv_writer = csv.writer(fout, delimiter="\t")
        for idx, emb in enumerate(ppmi_list):
            csv_writer.writerow([str(idx)] + emb + ['NONE'])

    feature_dict = {'entity_embedding': torch.tensor(ppmi_list, dtype=torch.float32)}
    if os.path.exists(output_path) == False:
        torch.save(feature_dict, output_path)
        

if __name__ == "__main__":
    main()
  