import os
import csv
import torch
import numpy as np


embed_files = [
    os.path.expanduser("~/QC-LinkPrediction-WIP/data/SEMNET/embeds/tsv/gemini_embedding.tsv)",
    os.path.expanduser("~/QC-LinkPrediction-WIP/data/SEMNET/embeds/tsv/Meta-Llama-3-70B_embedding.tsv)",
    os.path.expanduser("~/QC-LinkPrediction-WIP/data/SEMNET/embeds/tsv/Mixtral-8x7B-Instruct-v0.1_embedding.tsv)",
]
               
embed_list = []
for file in embed_files:
    with open(file) as fin:
        tsv_file = csv.reader(fin, delimiter="\t")
        
        embeds = []
        for line in tsv_file:
            embeds.append(line[1:-1])
            
        embed_list.append(embeds)

embed_list = np.array(embed_list, dtype=np.float32)

mean_pooled_embedding = np.mean(embed_list, axis=0)
max_pooled_embedding = np.max(embed_list, axis=0)

mean_pooled_embedding_list = mean_pooled_embedding.tolist()
max_pooled_embedding_list = max_pooled_embedding.tolist()

mean_pooled_keyword_embedding = []
max_pooled_keyword_embedding = []

for keyword_idx, (mean, max) in enumerate(zip(mean_pooled_embedding_list, max_pooled_embedding_list)):
    tab_separated_string = ""
    for item in mean:
        tab_separated_string += str(item) + "\t"
    mean_pooled_keyword_embedding.append(str(keyword_idx) + "\t" + tab_separated_string + "None")
    
    tab_separated_string = ""
    for item in max:
        tab_separated_string += str(item) + "\t"
    max_pooled_keyword_embedding.append(str(keyword_idx) + "\t" + tab_separated_string + "None")
    
output_path = os.path.expanduser("~/QC-LinkPrediction-WIP/data/SEMNET/embeds/tsv/mean_pooled_keyword_embedding.tsv)"
with open(output_path, 'w+') as fout:
    for i in mean_pooled_keyword_embedding:
        fout.write(i + '\n')
 
output_path = os.path.expanduser("~/QC-LinkPrediction-WIP/data/SEMNET/embeds/tsv/max_pooled_keyword_embedding.tsv)"
with open(output_path, 'w+') as fout:
    for i in max_pooled_keyword_embedding:
        fout.write(i + '\n')

output_path = os.path.expanduser("~/QC-LinkPrediction-WIP/data/SEMNET/pt/mean_pooled_embedding)"
feature_dict = {'entity_embedding': torch.tensor(mean_pooled_embedding, dtype=torch.float32)}
torch.save(feature_dict, output_path)

output_path = os.path.expanduser("~/QC-LinkPrediction-WIP/data/SEMNET/pt/max_pooled_embedding)"
feature_dict = {'entity_embedding': torch.tensor(max_pooled_embedding, dtype=torch.float32)}
torch.save(feature_dict, output_path)