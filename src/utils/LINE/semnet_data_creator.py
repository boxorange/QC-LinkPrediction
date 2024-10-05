import os

file = '/home/ac.gpark/QC-LinkPrediction-WIP/data/SEMNET/arxiv_qc_semnet_up_to_2021.tsv'

# line = [int(i) for i in l.replace("\n", "").split(" ")]

edge_with_weight = []

with open(file) as fin:
    for line in fin.readlines():
        n1, n2 = line.split()
        n1 = n1.strip()
        n2 = n2.strip()

        edge_with_weight.append([n1, n2, '1']) # set the weight to 1 as default.
        
with open('data/arxiv_qc_semnet_up_to_2021.edgelist', 'w') as fout:
    for item in edge_with_weight:
        fout.write(' '.join(item) + '\n')