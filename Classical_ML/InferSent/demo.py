#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import stuff
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from random import randint

import numpy as np
import torch


# ## Load model

# In[2]:


# Load model
from models import InferSent
model_version = 1
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))


# In[3]:


# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model


# In[4]:


# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)


# In[5]:


# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)


# ## Load sentences

# In[6]:


# Load some sentences
sentences = []
with open('samples.txt') as f:
    for line in f:
        sentences.append(line.strip())
print(len(sentences))


# In[7]:


sentences[:5]


# ## Encode sentences

# In[8]:


# gpu mode : >> 1000 sentences/s
# cpu mode : ~100 sentences/s


# In[9]:


embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
print('nb sentences encoded : {0}'.format(len(embeddings)))


# ## Visualization

# In[10]:


np.linalg.norm(model.encode(['the cat eats.']))


# In[11]:


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# In[12]:


cosine(model.encode(['the cat eats.'])[0], model.encode(['the cat drinks.'])[0])


# In[13]:


idx = randint(0, len(sentences))
_, _ = model.visualize(sentences[idx])


# In[14]:


my_sent = 'The cat is drinking milk.'
_, _ = model.visualize(my_sent)


# In[15]:


model.build_vocab_k_words(500000) # getting 500K words vocab
my_sent = 'barack-obama is the former president of the United-States.'
_, _ = model.visualize(my_sent)


# In[ ]:




