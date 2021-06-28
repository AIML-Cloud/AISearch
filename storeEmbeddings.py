from lang_model_utils import lm_vocab, load_lm_vocab, train_lang_model
from general_utils import save_file_pickle, load_file_pickle
data = list(train_df1.docstring_tokens)
type(data)
vocab = lm_vocab(max_vocab=50000,   min_freq=1)

# fit the transform on the training data, then transform
trn_flat_idx = vocab.fit_transform_flattened(data)
use_cache=False
if not use_cache:
    vocab.save('./data/lang_model/vocab_v2.cls')
    save_file_pickle('./data/lang_model/trn_flat_idx_list.pkl_v2', trn_flat_idx)
    learn.save('lang_model_learner_v2.fai')
    lang_model_new = learn.model.eval()   
    torch.save(lang_model_new.cpu(), './data/lang_model/lang_model_cpu_v2.torch')

vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')
trn_flat_idx = load_file_pickle('./data/lang_model/trn_flat_idx_list.pkl_v2')
from lang_model_utils import load_lm_vocab
vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')
idx_docs = vocab.transform(data, max_seq_len=30, padding=False)
item =[[2, 317, 3],[253, 1, 4]]

from time import sleep
from tqdm import tqdm
avg_hs, max_hs, last_hs = get_embeddings(lm_model, idx_docs)

savepath = Path('./data/lang_model_emb/')
np.save(savepath/'avg_emb_dim500_v2.npy', avg_hs)
np.save(savepath/'max_emb_dim500_v2.npy', max_hs)
np.save(savepath/'last_emb_dim500_v2.npy', last_hs)