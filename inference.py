from general_utils import create_nmslib_search_index
import nmslib
from lang_model_utils import Query2Emb
from pathlib import Path
import numpy as np
from lang_model_utils import load_lm_vocab
import torch
from lang_model_utils import lm_vocab, load_lm_vocab, train_lang_model
"""Fetching embeddings with 500 dimensions and loading it into the common 
vector space"""
loadpath = Path('./data/lang_model_emb/')
avg_emb_dim500 = np.load(loadpath/'avg_emb_dim500_v2.npy')
dim500_avg_searchindex = create_nmslib_search_index(avg_emb_dim500)
dim500_avg_searchindex.saveIndex('./data/lang_model_emb/dim500_avg_searchindex.nmslib')
dim500_avg_searchindex = nmslib.init(method='hnsw', space='cosinesimil')
dim500_avg_searchindex.loadIndex('./data/lang_model_emb/dim500_avg_searchindex.nmslib')

global V
%time
lang_model = torch.load('./data/lang_model/lang_model_cpu_v2.torch')
vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')

q2emb = Query2Emb(lang_model = lang_model.cpu(),
                  vocab = vocab)

class search_engine:
    def __init__(self, 
                 nmslib_index, 
                 ref_data, 
                 query2emb_func,df):
        
        self.search_index = nmslib_index
        self.data = ref_data
        self.df = df
        self.query2emb_func = query2emb_func
    
    def search(self, str_search, k=3):
        query = self.query2emb_func(str_search)
        idxs, dists = self.search_index.knnQuery(query, k=k)
        
        for idx, dist in zip(idxs, dists):
            print(f'cosine dist:{dist:.4f}\n---------------\n', self.data[idx])

se = search_engine(nmslib_index=dim500_avg_searchindex,
                   ref_data = df.docstring_tokens,
                   query2emb_func = q2emb.emb_mean,df=df)

query = 'Does Bexsero remain stable if it is exposed in sunglight for 20 hours at a temperature of 38 degrees ?'
se.search(query)