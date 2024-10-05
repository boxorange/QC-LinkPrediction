import os
import csv
import torch

model = torch.load('model.pt')

print(model.nodes_embeddings)
# print(model.nodes_embeddings[0])
# print(model.nodes_embeddings.shape)

# Get the embeddings
embeddings = model.nodes_embeddings.weight.data

# Detach the embeddings from the computation graph
embeddings = embeddings.detach()

line_pytorch_file_path = 'line_gnn_feature'

line_feature_dict = {'entity_embedding': torch.tensor(embeddings, dtype=torch.float32)}
if os.path.exists(line_pytorch_file_path) == False:
    torch.save(line_feature_dict, line_pytorch_file_path)

# Convert the embeddings to a list
embeddings_list = embeddings.tolist()

line_tsv_file_path = 'line_embedding.tsv'
with open(line_tsv_file_path, 'w') as fout:
    writer = csv.writer(fout, delimiter="\t")
    for idx, emb in enumerate(embeddings_list):
        writer.writerow([str(idx)] + emb + ['NONE'])


