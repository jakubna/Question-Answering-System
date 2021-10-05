#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json


# In[2]:


train = pd.read_json("data/train-v2.0.json")

# In[5]:


contexts = []
questions = []
ans_text = []
ans_start = []
for i in range(np.size(train, 0)):
    subject = train.iloc[i,1]['paragraphs']
    for paragraph in subject:
        for qa in paragraph['qas']:
            if len(qa['answers']) != 0:
                for answer in range(np.size(qa['answers'], 0)):
                    questions.append(qa['question'])
                    ans_start.append(qa['answers'][answer]['answer_start'])
                    ans_text.append(qa['answers'][answer]['text'])
                    contexts.append(paragraph['context'])  
data = pd.DataFrame({"context":contexts, "question": questions, "answer_start": ans_start, "text": ans_text})

# In[7]:


data.head(5)


# In[8]:


data.to_csv("data/train.csv", index = False)
del train

# In[9]:


paras = list(data["context"].drop_duplicates().reset_index(drop= True))


# In[10]:


from textblob import TextBlob
blob = TextBlob(" ".join(paras))
sentences = [item.raw for item in blob.sentences]


# In[14]:


questions = list(data["question"])


# In[15]:

del data
questions[0]


# In[16]:


from InferSent.models import InferSent
import torch
import nltk


# In[17]:


V = 1
MODEL_PATH = 'InferSent/encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = 'InferSent/GloVe/glove.840B.300d.txt'
infersent.set_w2v_path(W2V_PATH)


# In[18]:


infersent.build_vocab(sentences, tokenize=True)


# In[55]:
import pickle

# In[ ]:
print("embeddings building starting...")

embeddings_s = {}
for i in range(50000, 75000):
    embeddings_s[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)
    if i % 500 == 0:
        print(i)


# In[ ]:


pickle_out = open("data/emb_s_75k.pickle","wb")
pickle.dump(embeddings_s, pickle_out)
pickle_out.close()

del embeddings_s


# In[ ]:


embeddings_s = {}
for i in range(50000, len(sentences)):
    embeddings_s[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)
    if i % 500 == 0:
        print(i)


# In[ ]:


pickle_out = open("data/emb_s_rest.pickle","wb")
pickle.dump(embeddings_s, pickle_out)
pickle_out.close()

del embeddings_s
del sentences

# In[ ]:


embeddings_q = {}
for i in range(25000):
    embeddings_q[questions[i]] = infersent.encode([questions[i]], tokenize=True)
    if i % 500 == 0:
        print(i)


# In[ ]:


pickle_out = open("data/emb_q_25k.pickle","wb")
pickle.dump(embeddings_q, pickle_out)
pickle_out.close()

del embeddings_q


# In[ ]:


embeddings_q = {}
for i in range(25000, 50000):
    embeddings_q[questions[i]] = infersent.encode([questions[i]], tokenize=True)
    if i % 500 == 0:
        print(i)


# In[ ]:


pickle_out = open("data/emb_q_50k.pickle","wb")
pickle.dump(embeddings_q, pickle_out)
pickle_out.close()

del embeddings_q


# In[ ]:


embeddings_q = {}
for i in range(50000, 75000):
    embeddings_q[questions[i]] = infersent.encode([questions[i]], tokenize=True)
    if i % 500 == 0:
        print(i)


# In[ ]:


pickle_out = open("data/emb_q_75k.pickle","wb")
pickle.dump(embeddings_q, pickle_out)
pickle_out.close()

del embeddings_q


# In[ ]:


embeddings_q = {}
for i in range(75000, len(questions)):
    embeddings_q[questions[i]] = infersent.encode([questions[i]], tokenize=True)
    if i % 500 == 0:
        print(i)


# In[ ]:


pickle_out = open("data/emb_q_rest.pickle","wb")
pickle.dump(embeddings_q, pickle_out)
pickle_out.close()

del embeddings_q


# In[ ]:




