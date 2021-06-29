!pip install wget nmslib pathos ktext annoy
!https://github.com/fastai/fastai/blob/9a4a31d36f618db68d3c5250a40210b818cdc447/tests/test_vision_gan.py#L10
import torch,cv2
from lang_model_utils import lm_vocab, load_lm_vocab, train_lang_model
from general_utils import save_file_pickle, load_file_pickle
import logging
from pathlib import Path
from fastai.text import *
import numpy as np
# import ast
import glob
import re
from pathlib import Path

import astor
import pandas as pd
import spacy
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

from general_utils import apply_parallel, flattenlist
from ktext.preprocess import processor
import dill as dpickle
import numpy as np
from seq2seq_utils import load_decoder_inputs, load_encoder_inputs, load_text_processor
from seq2seq_utils import build_seq2seq_model
from keras.models import Model, load_model
import pandas as pd
import logging
from keras.callbacks import CSVLogger, ModelCheckpoint
import numpy as np
from keras import optimizers

from seq2seq_utils import Seq2Seq_Inference
import pandas as pd
from general_utils import get_step2_prerequisite_files
import torch,cv2
from lang_model_utils import lm_vocab, load_lm_vocab, train_lang_model
from general_utils import save_file_pickle, load_file_pickle
import logging
from pathlib import Path
from fastai.text import *
import numpy as np
EN = spacy.load('en_core_web_sm')

import get_Embeddings

df = pd.read_csv('/content/dummydummy.csv',encoding='Latin-1')
print(df.shape)
print(df.columns)

from fastai.text import * 

## stemming 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

porter=PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


train_df1 = df.head(199)
valid_df1 = df.tail(90)
"""
Wraping an iterable on the imported dataset and Checking the converted fastai dataset"""
data_lm = TextLMDataBunch.from_df(train_df = train_df1, valid_df = valid_df1, text_cols='docstring_tokens', path = "")

data_lm.show_batch()


def config(qrnn:bool=False):
    config = awd_lstm_lm_config.copy()
    config['n_hid'] = 1152
    return config

"""
Training AWD_LSTM model instead of TransformerXL and Transformer as the dataset 
is having documents with lots of unstructured text"""
learn = language_model_learner(data_lm, AWD_LSTM, config=config(),drop_mult=0.5)
#learn = language_model_learner(data_lm, TransformerXL,drop_mult=0.5)
# learn = language_model_learner(data_lm, Transformer,drop_mult=0.5)

learn.fit_one_cycle(3, 1e-2)

from tensorboardX import SummaryWriter
from fastai.vision import * 
from fastai.text import * 
from fastai.collab import *
from fastai.callbacks.tensorboard import *

project_id = 'project'
tboard_path = Path('./' + project_id)
learn.callback_fns.append(partial(LearnerTensorboardWriter, 
                                    base_dir=tboard_path, 
                                    name='run1'))
learn.fit(1, 1e-2)
learn.model.eval()

