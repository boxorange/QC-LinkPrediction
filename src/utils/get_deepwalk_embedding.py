import networkx as nx
from gensim.models import Word2Vec
import random
import torch


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, dimensions, window_size, workers=1):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = workers

    def generate_random_walks(self):
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(node))
        return walks

    def random_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur_node = walk[-1]
            neighbors = list(self.graph.neighbors(cur_node))
            if len(neighbors) > 0:
                next_node = random.choice(neighbors)
                walk.append(next_node)
            else:
                break
        return walk

    def train(self):
        walks = self.generate_random_walks()
        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(sentences=walks, vector_size=self.dimensions, window=self.window_size, min_count=0, sg=1, workers=self.workers)
        return model

def load_graph_from_edge_list(edge_list_file, isolated_nodes=[]):
    graph = nx.Graph()
    with open(edge_list_file, 'r') as f:
        for line in f:
            edge = line.strip().split('\t')
            if len(edge) == 2:
                graph.add_edge(edge[0], edge[1])
    for node in isolated_nodes:
        graph.add_node(node)
    return graph

def save_embeddings(model, file_path, device):
    embeddings = {}
    for node in model.wv.index_to_key:
        embeddings[node] = torch.tensor(model.wv[node], device=device)
    
    # Sort dictionary by keys
    sorted_embeddings = {k: embeddings[k] for k in sorted(embeddings)}

    # Convert dictionary values to 2D tensor
    # tensor_values = torch.tensor([v for v in sorted_embeddings.values()])
    
    # Convert dictionary values to 2D tensor
    tensor_values = torch.stack([v for v in sorted_embeddings.values()])

    # Save tensor to a torch file
    torch.save(tensor_values, file_path)

        
if __name__ == "__main__":
    # Path to the edge list file
    edge_list_file = '~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_up_to_2021.tsv'
    edge_list_file = os.path.expanduser(edge_list_file)
    
    kw_file = '~/QC-LinkPrediction/data/SEMNET/arxiv_qc_semnet_keywords_2024.txt'
    kw_file = os.path.expanduser(kw_file)
    
    kw_ids = set()
    with open(kw_file, "r") as fin:
        for line in fin.readlines():
            idx, kw = line.split('\t')
            kw = kw.strip()
            kw_ids.add(idx)
    
    edge_kw_ids = set()
    with open(edge_list_file, "r") as fin:
        for line in fin.readlines():
            id1, id2 = line.split('\t')
            id1 = id1.strip()
            id2 = id2.strip()
            edge_kw_ids.add(id1)
            edge_kw_ids.add(id2)
                        
    # List of isolated nodes
    isolated_nodes = kw_ids.difference(edge_kw_ids)
    isolated_nodes = list(isolated_nodes)

    # Load graph from edge list
    G = load_graph_from_edge_list(edge_list_file, isolated_nodes)

    # Initialize DeepWalk
    # deepwalk = DeepWalk(graph=G, walk_length=10, num_walks=80, dimensions=768, window_size=5, workers=4)
    deepwalk = DeepWalk(graph=G, walk_length=40, num_walks=80, dimensions=768, window_size=10, workers=4)

    # Train model and generate embeddings
    model = deepwalk.train()

    # Specify the device (CPU)
    device = torch.device('cpu')

    # Save embeddings to a torch file
    save_embeddings(model, 'semnet-deepwalk-embedding.pt', device)

    print(f"Embeddings saved to 'semnet-deepwalk-embedding.pt'")
