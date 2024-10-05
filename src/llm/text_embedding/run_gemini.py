"""
refs: https://ai.google.dev/gemini-api/docs/models/gemini#text-embedding
      https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#generative-ai-get-text-embedding-python_vertex_ai_sdk
      https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb
      
"""

import google.generativeai as genai
import os
import argparse
import csv
import pickle
import torch
import time
from itertools import islice
from datetime import timedelta, datetime


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

parser = argparse.ArgumentParser()

# general arguments
parser.add_argument('--data_dir', type=str, default='', required=True)
parser.add_argument('--output_dir', type=str, default='', required=True)
parser.add_argument('--keywords_path', type=str, default='', required=True)
parser.add_argument('--task_type', type=str, default='query')
parser.add_argument('--embed_dim', type=int, default=768)
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()

data_dir = os.path.expanduser(args.data_dir)
output_dir = os.path.expanduser(args.output_dir)
keywords_path = os.path.expanduser(args.keywords_path)
task_type = args.task_type
embed_dim = args.embed_dim
batch_size = args.batch_size

output_dir = os.path.join(output_dir, 'gemini_task_type_' + task_type)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# this is used to match indices in the keyword file.
selected_keywords = {}
with open(keywords_path) as fin:
    for line in fin.readlines():
        idx, kw = line.split('\t')
        selected_keywords[kw.strip()] = idx

for filename in os.listdir(data_dir):
    output_path = os.path.join(output_dir, filename.replace('keyword_feature.pickle', 'embedding.tsv'))

    if os.path.exists(output_path):
        print(f'>> the embedding file already exists: {output_path}')
        continue
    
    # run a model.
    st = time.time()

    with open(os.path.join(data_dir, filename), 'rb') as fin:
        keyword_feature = pickle.load(fin)
    
    # this is used to filter keywords from the existing feature files. e.g., semnet_keywords_updated -> arxiv_qc_semnet_keywords_2017
    keywords_to_be_ignored = set(keyword_feature) - set(list(selected_keywords.keys()))
    for kw in keywords_to_be_ignored:
        if kw in keyword_feature:
            del keyword_feature[kw]

    keyword_embedding = []

    def chunks(dict, batch_size=16):
        it = iter(dict)
        for i in range(0, len(dict), batch_size):
            yield {k: dict[k] for k in islice(it, batch_size)}

    for batch in chunks(keyword_feature, batch_size):
        keywords = list(batch.keys())
        
        
        
        
        ## TODO: temporary code. delete this later. 
        if filename == 'gemini_arxiv_qc_sum_keyword_feature.pickle':
            features = [x['summary'] for x in list(batch.values())]
        else:
            features = list(batch.values())




        result = genai.embed_content(
            # model="models/embedding-001", # old model
            model="models/text-embedding-004",
            content=features,
            task_type=task_type, # "query" (default), "document", "semantic_similarity", "classification", "clustering", "question_answering", "fact_verification"
            output_dimensionality=embed_dim, # output_dimensionality: Optional. Reduced dimension for the output embedding. If set, excessive values in the output embedding are truncated from the end. This is supported by models/text-embedding-004, but cannot be specified in models/embedding-001.
        )
        
        assert len(keywords) == len(features) == len(result['embedding']), "The lists do not have the same length!"


        # print(result)
        # input('enter..')
        
        
        for kw, feat, embed in zip(keywords, features, result['embedding']):
            keyword_idx = selected_keywords[kw]
            '''
            tab_separated_string = ""
            for val in embed:
                tab_separated_string += str(val) + "\t"
            keyword_embedding.append(keyword_idx + "\t" + tab_separated_string + "None")
            '''
            embed.insert(0, keyword_idx)
            keyword_embedding.append(embed)
            
            
    '''	
    for idx, (k, v) in enumerate(keyword_feature.items()):
        result = genai.embed_content(
            model="models/embedding-001",
            content=v,
        )
        
        embedding = result['embedding']

        tab_separated_string = ""
        for item in embedding:
            tab_separated_string += str(item) + "\t"
            
        keyword_embedding.append(str(idx) + "\t" + tab_separated_string + "None")
        
        #print(embedding)
        #for i in keyword_embedding:
        #	print(i)
        #input('enter..')
    '''
    
    # keyword_embedding = sorted(keyword_embedding, key=lambda x: int(x.split('\t')[0]))
    keyword_embedding = sorted(keyword_embedding, key=lambda x: int(x[0]))
    
    # with open(output_path, 'w+') as fout:
        # for i in keyword_embedding:
            # fout.write(i + '\n')
    
    with open(output_path, 'w') as fout:
        csv_writer = csv.writer(fout, delimiter="\t")
        for emb in keyword_embedding:
            csv_writer.writerow(emb + ['NONE'])
    
    keyword_embedding = [x[1:] for x in keyword_embedding] # leave out indices.

    feature_dict = {'entity_embedding': torch.tensor(keyword_embedding, dtype=torch.float32)}
    
    output_path = output_path.replace('.tsv', '')
    if os.path.exists(output_path) == False:
        torch.save(feature_dict, output_path)

    et = time.time()
    elapsed_time = et - st
    exec_time = timedelta(seconds=elapsed_time)
    exec_time = str(exec_time)
    print('>> Execution time in hh:mm:ss:', exec_time)

