import os.path as osp
from typing import Callable, List, Optional, Dict

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import coalesce


class Quantum(InMemoryDataset):
    def __init__(
        self, 
        root: str, 
        name: str,
        raw_data_paths: Dict[str, str] = None,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        assert self.name in ['semnet']
        
        self.raw_data_paths = raw_data_paths
        
        super().__init__(root, transform, pre_transform)

        if split == 'train':
            self.load(self.processed_paths[0])
        elif split == 'valid':
            self.load(self.processed_paths[1])
        elif split == 'test':
            self.load(self.processed_paths[2])

    # @property
    # def raw_dir(self) -> str:
        # return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        # return osp.join(self.root, self.name, 'processed')
        return self.raw_data_paths['processed_dir']


    @property
    def raw_file_names(self) -> List[str]:
        # names = ['yr_now_concept_2012.tsv', 'yr_now_cooccurrence_2012_train.tsv', 'yr_now_cooccurrence_2012_val.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['yr_now_concept_2012.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['none_keyword_embedding.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['palm_keyword_embedding.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['gemini_keyword_embedding.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['Llama-2-70b-chat-hf_keyword_embedding.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['mpt-30b-chat_keyword_embedding.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['Mixtral-8x7B-Instruct-v0.1_keyword_embedding.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['mean_pooled_keyword_embedding.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['max_pooled_keyword_embedding.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['gemini_sum_keyword_embedding.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # return [f'{name}.npz' for name in names]
        
        
        # names = ['gemini_keyword_embedding.tsv', 'yr_now_cooccurrence_2012_train.tsv', 'yr_now_cooccurrence_2012_val.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['gemini_keyword_embedding.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['none_keyword_embedding.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['llm_blender_keyword_embedding_4.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['Meta-Llama-3-70B_keyword_embedding_4.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        # names = ['gemini_time_decay_keyword_embedding.tsv', 'yr_now_cooccurrence_2012.tsv', 'cooccurrence_2012_2017.tsv']
        
        names = [self.raw_data_paths['embed_path'], self.raw_data_paths['train_data_path'], self.raw_data_paths['valid_data_path'], self.raw_data_paths['test_data_path']]
        
        
        return names

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'valid_data.pt', 'test_data.pt']
        # return ['train_data.pt', 'test_data.pt']
        # return ['now_data.pt', 'delta_data.pt']
    
    '''
    def download(self) -> None:
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/{self.name}/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)
    '''
    
    def process(self) -> None:
        with open(self.raw_paths[0], 'r') as f:
            lines = f.read().split('\n')[:-1]
            xs = [[float(value) for value in line.split('\t')[1:-1]] for line in lines]
            x = torch.tensor(xs, dtype=torch.float)

            ys = [0 for line in lines] # set to 0 for no labels.
            y = torch.tensor(ys, dtype=torch.long)
        
        for num, file in enumerate(self.raw_paths[1:]):
            with open(file, 'r') as f:
                lines = f.read().split('\n')[:-1]
                edge_indices = [[int(value) for value in line.split('\t')] for line in lines]
                edge_index = torch.tensor(edge_indices).t().contiguous()
                edge_index = coalesce(edge_index, num_nodes=x.size(0))
                
                data = Data(x=x, edge_index=edge_index, y=y)
                data = data if self.pre_transform is None else self.pre_transform(data)
                self.save([data], self.processed_paths[num])

    def __repr__(self) -> str:
        return f'{self.name}()'