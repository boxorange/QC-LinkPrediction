import os
import json
import argparse
import random
import copy
from itertools import combinations
from collections import Counter


def find_keyword_cooccurrences(documents, keywords):
    # Initialize a Counter to hold the co-occurrences
    cooccurrences = Counter()
    
    kw_counter = Counter()
    num_of_docs_having_pair_of_concepts = []
    
    # Iterate over each document
    for idx, doc in enumerate(documents):
        # Find all the keywords present in the document
        present_keywords = [keyword for keyword in keywords if keyword in doc]
        
        for pk in present_keywords:
            kw_counter[pk] += 1
        
        if len(present_keywords) >= 2:
            num_of_docs_having_pair_of_concepts.append(idx)
        
        # Get all combinations of two keywords that are present in the document
        for combo in combinations(present_keywords, 2):
            # Update the count for this combination in the Counter
            cooccurrences[combo] += 1

    return cooccurrences


def generate_cooccurrence_counter(corpus, start_year, end_year, keywords):
    cooccurrence_by_year = {}

    for year, docs in corpus.items():
        if int(year) >= start_year and int(year) <= end_year:		
            documents = [x['title'] + ' ' + x['abstract'] for x in docs]
            counter = find_keyword_cooccurrences(documents, keywords.values())
            cooccurrence_by_year[year] = counter

    return cooccurrence_by_year


def generate_data(
    corpus_file_path,
    orig_semnet_concept_file_path,
    arxiv_semnet_concept_file_path,
    output_dir,
    train_data_path,
    valid_data_path,
    valid_test_ratio,
    start_year,
    end_year,
):
    # read the corpus. 
    corpus = json.load(open(corpus_file_path)) 

    # read the SEMNET keywords existing in the arXiv corpus.
    if os.path.exists(arxiv_semnet_concept_file_path):
        arxiv_semnet_keyword_file = open(arxiv_semnet_concept_file_path)
        arxiv_semnet_keywords = {}
        for x in arxiv_semnet_keyword_file.readlines():
            id, keyword = x.split('\t', 1)
            arxiv_semnet_keywords[int(id.strip())] = keyword.strip()
    else:
        # to create a list of SEMNET keywords existing in arXiv, first check what SEMNET keywords of the original list in the arXiv corpus. 
        # read the original SEMNET keywords.
        orig_semnet_keyword_file = open(orig_semnet_concept_file_path)
        orig_semnet_keywords = {}
        for x in orig_semnet_keyword_file.readlines():
            id, keyword = x.split('\t', 1)
            orig_semnet_keywords[int(id.strip())] = keyword.strip()

        cooccurrence_by_year = generate_cooccurrence_counter(corpus, start_year, end_year, orig_semnet_keywords)
            
        keywords_in_arxiv = set()

        for year, cooccurrence in cooccurrence_by_year.items():
            for combo, cnt in cooccurrence.items():
                keywords_in_arxiv.update(combo)

        keywords_in_arxiv = sorted(list(keywords_in_arxiv))

        with open(arxiv_semnet_concept_file_path, 'w') as fout:
            for index, keyword in enumerate(keywords_in_arxiv):
                fout.write(f"{index}\t{keyword}\n")
                
        arxiv_semnet_keywords = {id: keyword for id, keyword in enumerate(keywords_in_arxiv)}
    
    cooccurrence_by_year = generate_cooccurrence_counter(corpus, start_year, end_year, arxiv_semnet_keywords)
    
    unique_cooccurrences = set()
    for cooccurrence in cooccurrence_by_year.values():
        for combo in cooccurrence.keys():
            unique_cooccurrences.add(tuple(sorted(combo)))
    
    if start_year == 2007: # 2007 is the beginning year of the corpus.
        output_file = f"arxiv_qc_semnet_up_to_{end_year}.tsv"
    else:
        output_file = f"arxiv_qc_semnet_between_{start_year}_and_{end_year}.tsv"
    
    arxiv_semnet_kw2idx = {value: key for key, value in arxiv_semnet_keywords.items()}
    
    if train_data_path != None:
        train_set = set()
        with open(train_data_path) as fin:
            lines = fin.readlines()
            
        for line in lines:
            train_set.add(tuple([arxiv_semnet_keywords[int(x.strip())] for x in line.split('\t', 1)]))
        
        unique_train_keywords = set()
        for kw1, kw2 in train_set:
            unique_train_keywords.add(kw1)
            unique_train_keywords.add(kw2)

        unique_cooccurrences.difference_update(train_set)
        
        if valid_data_path != None:
            valid_set = set()
            with open(valid_data_path) as fin:
                lines = fin.readlines()
                
            for line in lines:
                valid_set.add(tuple([arxiv_semnet_keywords[int(x.strip())] for x in line.split('\t', 1)]))

            unique_cooccurrences.difference_update(valid_set)

        unique_val_or_test_keywords = set()
        for kw1, kw2 in unique_cooccurrences:
            unique_val_or_test_keywords.add(kw1)
            unique_val_or_test_keywords.add(kw2)
            
        unique_val_or_test_keywords.difference_update(unique_train_keywords)

        # If it's smaller than valid_test_ratio, then use the entire data.
        data_size = int(len(train_set) * valid_test_ratio)
        data_size = data_size if data_size <= len(unique_cooccurrences) else len(unique_cooccurrences)

        # randomly select samples. 
        # the selected samples must contain all keywords appearing exclusively in the validation or test data and not in the training data.
        # this is necessary in a transductive graph setting where all training, validation, and test graphs share the same nodes.
        unique_cooccurrences = list(unique_cooccurrences) # DeprecationWarning: Sampling from a set deprecated since Python 3.9 and will be removed in a subsequent version.
        SEED = 1
        while True:
            random.seed(SEED)
            selected_samples = set(random.sample(unique_cooccurrences, data_size))
            
            selected_samples_keywords = set()
            for kw1, kw2 in selected_samples:
                selected_samples_keywords.add(kw1)
                selected_samples_keywords.add(kw2)
            
            if unique_val_or_test_keywords.issubset(selected_samples_keywords):
                break
            
            # for residual keywords, add all pairs containing any of these keywords to the sample.
            residual_keywords = copy.deepcopy(unique_val_or_test_keywords)
            residual_keywords.difference_update(selected_samples_keywords)
            residual_set = set([t for t in unique_cooccurrences if any(k in t for k in residual_keywords)])

            # Remove the number of elements equal to the size of the new set
            if len(residual_set) <= len(selected_samples):
                for _ in range(len(residual_set)):
                    selected_samples.pop()
            else:
                print("The existing set does not have enough elements to remove.")

            # Append the new set to the existing set
            selected_samples.update(residual_set)
            
            # re-check if the selected samples contain all keywords appearing exclusively in the validation or test data.
            selected_samples_keywords = set()
            for kw1, kw2 in selected_samples:
                selected_samples_keywords.add(kw1)
                selected_samples_keywords.add(kw2)
            
            if unique_val_or_test_keywords.issubset(selected_samples_keywords):
                break
                
            # for residual keywords, add all pairs containing any of these keywords to the sample.
            residual_keywords = copy.deepcopy(unique_val_or_test_keywords)
            residual_keywords.difference_update(selected_samples_keywords)
            residual_set = set([t for t in unique_cooccurrences if any(k in t for k in residual_keywords)])
    
            SEED += 1

        unique_cooccurrences = selected_samples

    with open(os.path.join(output_dir, output_file), 'w+') as fout:
        for pair in unique_cooccurrences:
            fout.write(f"{arxiv_semnet_kw2idx[pair[0]]}\t{arxiv_semnet_kw2idx[pair[1]]}\n")

    print(f'<generate_data()> Done. The number of edges: {len(unique_cooccurrences)}')
    

def main():
    parser = argparse.ArgumentParser()
    
    # general args
    parser.add_argument('--corpus_file_path', type=str, required=True, help="quantum physics corpus file from arXiv.")
    parser.add_argument('--orig_semnet_concept_file_path', type=str, required=True, help="SEMNET concept list file.")
    parser.add_argument('--arxiv_semnet_concept_file_path', type=str, required=True, help="SEMNET concept list file existing in arXiv corpus.")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--train_data_path', type=str, default='')
    parser.add_argument('--valid_data_path', type=str, default='')
    parser.add_argument('--valid_test_ratio', type=float, default=0.125)
    parser.add_argument('--start_year', type=int, default=2007)
    parser.add_argument('--end_year', type=int, default=2012)

    args = parser.parse_args()

    corpus_file_path = args.corpus_file_path
    orig_semnet_concept_file_path = os.path.expanduser(args.orig_semnet_concept_file_path)
    arxiv_semnet_concept_file_path = os.path.expanduser(args.arxiv_semnet_concept_file_path)
    output_dir = os.path.expanduser(args.output_dir)
    train_data_path = os.path.expanduser(args.train_data_path) if args.train_data_path != '' else None
    valid_data_path = os.path.expanduser(args.valid_data_path) if args.valid_data_path != '' else None
    valid_test_ratio = args.valid_test_ratio
    start_year = args.start_year
    end_year = args.end_year

    generate_data(
        corpus_file_path,
        orig_semnet_concept_file_path,
        arxiv_semnet_concept_file_path,
        output_dir,
        train_data_path,
        valid_data_path,
        valid_test_ratio,
        start_year,
        end_year,
    )
    

if __name__ == "__main__":
    main()
