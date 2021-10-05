#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle


# In[ ]:


data = pd.read_csv("data/train.csv")


# ## Open InferSent embeddings and create dictionaries

# In[3]:


pickle_in = open("data/emb_s_25k.pickle", "rb")
sen1 = pickle.load(pickle_in)
pickle_in.close()


# In[4]:


pickle_in = open("data/emb_s_50k.pickle", "rb")
sen2 = pickle.load(pickle_in)
pickle_in.close()


# In[5]:


pickle_in = open("data/emb_s_75k.pickle", "rb")
sen3 = pickle.load(pickle_in)
pickle_in.close()


# In[6]:


pickle_in = open("data/emb_s_rest.pickle", "rb")
sen4 = pickle.load(pickle_in)
pickle_in.close()


# In[7]:


dict_sen = dict(sen1)
del sen1


# In[8]:


dict_sen.update(sen2)
del sen2


# In[9]:


dict_sen.update(sen3)
del sen3


# In[10]:


dict_sen.update(sen4)
del sen4


# In[11]:


pickle_in = open("data/emb_q_25k.pickle", "rb")
que1 = pickle.load(pickle_in)
pickle_in.close()


# In[12]:


pickle_in = open("data/emb_q_50k.pickle", "rb")
que2 = pickle.load(pickle_in)
pickle_in.close()


# In[13]:


pickle_in = open("data/emb_q_75k.pickle", "rb")
que3 = pickle.load(pickle_in)
pickle_in.close()


# In[14]:


pickle_in = open("data/emb_q_rest.pickle", "rb")
que4 = pickle.load(pickle_in)
pickle_in.close()


# In[15]:


dict_que = dict(que1)
del que1


# In[16]:


dict_que.update(que2)
del que2


# In[17]:


dict_que.update(que3)
del que3


# In[18]:


dict_que.update(que4)
del que4


# ## Add splitted contexts, answers and embeddings to the Data Frame

# In[19]:


from textblob import TextBlob
import nltk
nltk.download('punkt')


# In[20]:


def split_context(context):
    blob = TextBlob(context)
    sentences = [item.raw for item in blob.sentences]
    return sentences


# In[21]:


def find_answer(qa):
    for i in range(len(qa["sentences"])):
        if not isinstance(qa["text"], str):
            qa["text"] = str(qa["text"])
        if qa["text"] in qa["sentences"][i]:
            return i
    return -1


# In[22]:


data["sentences"] = data["context"].apply(split_context)
data["ans_sentence"] = data.apply(find_answer, axis = 1)


# In[23]:


def get_que_emb(question):
    if question in dict_que:
        return dict_que[question]
    return np.zeros(4096)


# In[24]:


import numpy as np

def get_sent_emb(sentences):
    res = [dict_sen[item][0] if item in dict_sen else np.zeros(4096) for item in sentences]
    return res


# In[25]:


data["que_emb"] = data["question"].apply(get_que_emb)
data["sent_emb"] = data["sentences"].apply(get_sent_emb)


# In[26]:


del dict_que
del dict_sen


# In[27]:


del data["context"]


# In[28]:


data.head(15)


# ## Count Euclidean and cosine distances between the question and each answer

# In[29]:


from scipy import spatial

def eucl_dis(record):
    res = [0] * len(record["sent_emb"])
    for i in range(len(record["sent_emb"])):
        res[i] = spatial.distance.euclidean(record["sent_emb"][i],record["que_emb"][0])
    return res


# In[30]:


def cosine_dis(record):
    res = [0]*len(record["sent_emb"])
    for i in range(len(record["sent_emb"])):
        res[i] = spatial.distance.cosine(record["sent_emb"][i],record["que_emb"][0])
    return res


# In[34]:


data["eucl_dis"] = data.apply(eucl_dis, axis = 1)


# In[35]:


data["cosine_dis"] = data.apply(cosine_dis, axis = 1)


# In[36]:


data.head(10)


# In[38]:


data.to_csv("data/processed.csv", index=None)

