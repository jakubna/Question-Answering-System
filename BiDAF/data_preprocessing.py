#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import random
import argparse
import json
import nltk
import numpy as np
from tqdm import tqdm
from six.moves.urllib.request import urlretrieve


# In[2]:


def data_from_json(filename):
    """Loads JSON data from filename and returns"""
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


# In[3]:


def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]
    return tokens


# In[4]:


def write_to_file(out_file, line):
    out_file.write(str(line.encode('utf8')) + str('\n'))


# In[5]:


def get_char_word_loc_mapping(context, context_tokens):
    """
    Return a mapping that maps from character locations to the corresponding token locations.
    If we're unable to complete the mapping e.g. because of special characters, we return None.

    Inputs:
      context: string (unicode)
      context_tokens: list of strings (unicode)

    Returns:
      mapping: dictionary from ints (character locations) to (token, token_idx) pairs
        Only ints corresponding to non-space character locations are in the keys
        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
        0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
    """
    acc = '' # accumulator
    current_token_idx = 0 # current word loc
    mapping = dict()

    for char_idx, char in enumerate(context): # step through original characters
        if char != u' ' and char != u'\n': # if it's not a space:
            acc += char # add to accumulator
            context_token = context_tokens[current_token_idx] # current word token
            if acc == context_token: # if the accumulator now matches the current word token
                syn_start = char_idx - len(acc) + 1 # char loc of the start of this word
                for char_loc in range(syn_start, char_idx+1):
                    mapping[char_loc] = (acc, current_token_idx) # add to mapping
                acc = '' # reset accumulator
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping


# In[6]:


def preprocess_and_write(dataset, tier, out_dir):
    
    num_exs = 0 # number of examples written to file
    num_mappingprob, num_tokenprob, num_spanalignprob, num_noanswer = 0, 0, 0, 0
    examples = []
    
    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):

        article_paragraphs = dataset['data'][articles_id]['paragraphs']

        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context'] # string
            # The following replacements are suggested in the paper

            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context) # list of strings (lowercase)
            context = context.lower()

            qas = article_paragraphs[pid]['qas'] # list of questions


            charloc2wordloc = get_char_word_loc_mapping(context, context_tokens) # charloc2wordloc maps the character location (int) of a context token to a pair giving (word (string), word loc (int)) of that token
            if charloc2wordloc is None: # there was a problem
                num_mappingprob += len(qas)
                continue # skip this context example

            # for each question, process the question and answer and write to file
            for qn in qas:
                try:
                    # read the question text and tokenize
                    question = qn['question'] # string
                    question_tokens = tokenize(question) # list of strings

                    # of the three answers, just take the first
                    try:
                        ans_text = qn['answers'][0]['text'].lower() # get the answer text
                        ans_start_charloc = qn['answers'][0]['answer_start'] # answer start loc (character count)
                    except:
                        ans_text = qn['plausible_answers'][0]['text'].lower() # get the answer text
                        ans_start_charloc = qn['plausible_answers'][0]['answer_start'] # answer start loc (character count)

                    ans_end_charloc = ans_start_charloc + len(ans_text) # answer end loc (character count) (exclusive)

                    # Check that the provided character spans match the provided answer text
                    if context[ans_start_charloc:ans_end_charloc] != ans_text:
                      # Sometimes this is misaligned, mostly because "narrow builds" of Python 2 interpret certain Unicode characters to have length 2 https://stackoverflow.com/questions/29109944/python-returns-length-of-2-for-single-unicode-character-string
                      # We should upgrade to Python 3 next year!
                      num_spanalignprob = num_spanalignprob + 1
                      continue

                    # get word locs for answer start and end (inclusive)
                    ans_start_wordloc = charloc2wordloc[ans_start_charloc][1] # answer start word loc
                    ans_end_wordloc = charloc2wordloc[ans_end_charloc-1][1] # answer end word loc
                    assert ans_start_wordloc <= ans_end_wordloc

                    # Check retrieved answer tokens match the provided answer text.
                    # Sometimes they won't match, e.g. if the context contains the phrase "fifth-generation"
                    # and the answer character span is around "generation",
                    # but the tokenizer regards "fifth-generation" as a single token.
                    # Then ans_tokens has "fifth-generation" but the ans_text is "generation", which doesn't match.
                    ans_tokens = context_tokens[ans_start_wordloc:ans_end_wordloc+1]
                    if "".join(ans_tokens) != "".join(ans_text.split()):
                        num_tokenprob += 1
                        continue # skip this question/answer pair
                    
                    examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(ans_tokens), ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)])))
                    num_exs += 1
                except:
                    num_noanswer += 1

    print ("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", num_mappingprob)
    print ("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_tokenprob)
    print ("Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): ", num_spanalignprob)
    print ("Number of questions discarded due to not existing answer: ", num_noanswer)
    print ("Processed %i examples of total %i\n" % (num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalignprob))

    
    # shuffle examples
    indices = list(range(len(examples)))
    np.random.shuffle(indices)

    out_dir = './data/preprocessed'

    with open(os.path.join(out_dir, tier +'.context'), 'w') as context_file,           open(os.path.join(out_dir, tier +'.question'), 'w') as question_file,         open(os.path.join(out_dir, tier +'.answer'), 'w') as ans_text_file,          open(os.path.join(out_dir, tier +'.span'), 'w') as span_file:

        for i in indices:
            (context, question, answer, answer_span) = examples[i]

            # write tokenized data to file
            write_to_file(context_file, str(context))
            write_to_file(question_file, str(question))
            write_to_file(ans_text_file, str(answer))
            write_to_file(span_file, str(answer_span))


# In[7]:


import nltk
nltk.download('punkt')

train_filepath = './data/train-v2.0.json'
dev_filepath = './data/dev-v2.0.json'

preprocess_and_write(data_from_json(train_filepath), 'train', './data')
preprocess_and_write(data_from_json(dev_filepath), 'dev', './data')


# In[8]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import argparse

from six.moves import urllib

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin

_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2


# In[9]:


def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("data", "squad")
    glove_dir = os.path.join("download", "dwr")
    source_dir = os.path.join("data", "squad")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=100, type=int)
    parser.add_argument("--random_init", default=True, type=bool)
    return parser.parse_args()


# In[10]:


def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


# In[11]:


def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


# In[12]:


def process_glove(vocab_list, size=4e5, random_init=True):

    save_path = './glove_vectors/_vectors'
    glove_dim = 200
    
    if not gfile.Exists(save_path + ".npz"):
        glove_path = './glove_vectors/glove.6B.200d.txt'
        if random_init:
            glove = np.random.randn(len(vocab_list), glove_dim)
        else:
            glove = np.zeros((len(vocab_list), glove_dim))
        found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                    found += 1

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


# In[13]:


def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
#     if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
    vocab = {}
    for path in data_paths:
        with open(path, mode="rb") as f:
            print(f)
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("processing line %d" % counter)
                tokens = tokenizer(str(line)) if tokenizer else basic_tokenizer(str(line))
                for w in tokens:
                    if w in vocab:
                        vocab[w] += 1
                    else:
                        vocab[w] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    print("Vocabulary size: %d" % len(vocab_list))
    with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
            vocab_file.write(str(w) + "\n")


# In[14]:


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    if tokenizer:
        words = tokenizer(str(sentence))
    else:
        words = basic_tokenizer(str(sentence))
    return [vocabulary.get(w, UNK_ID) for w in words]


# In[15]:


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 5000 == 0:
                        print("tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


# In[16]:


source_dir = './data/preprocessed'
vocab_dir = './vocabulary'

vocab_path = pjoin(vocab_dir, "vocab.dat")
train_path = pjoin(source_dir, "train")
valid_path = pjoin(source_dir, "dev")
dev_path = pjoin(source_dir, "dev")


create_vocabulary(vocab_path,
                  [pjoin(source_dir, "train.context"),
                   pjoin(source_dir, "train.question")
                  ])


# In[17]:


vocab, rev_vocab = initialize_vocabulary(pjoin(vocab_dir, "vocab.dat"))


# In[18]:


# ======== Trim Distributed Word Representation =======
# If you use other word representations, you should change the code below

process_glove(rev_vocab, random_init=0)


# In[19]:


import sys
import codecs

file_path_train_span = '/home/andy/Documents/Moje/WEDT/projekt/my_code/data/preprocessed/train.span'
file_path_dev_span = '/home/andy/Documents/Moje/WEDT/projekt/my_code/data/preprocessed/dev.span'

def clear_span(file_path):
    f = codecs.open(file_path_train_span, encoding='utf-8')
    contents = f.read()


    newcontents = contents.replace('b','')
    newcontents = newcontents.replace('\'', '')
#     print(newcontents)

    f.close()
    
    x=open(file_path,"w")
    x.write(newcontents)
    x.close
    
clear_span(file_path_train_span)
clear_span(file_path_dev_span)


# In[20]:


# ======== Creating Dataset =========
# We created our data files seperately
# If your model loads data differently (like in bulk)
# You should change the below code

x_train_ids_path = train_path + ".ids.context"
y_train_ids_path = train_path + ".ids.question"
data_to_token_ids(train_path + ".context", x_train_ids_path, vocab_path)
data_to_token_ids(train_path + ".question", y_train_ids_path, vocab_path)

x_dis_path = valid_path + ".ids.context"
y_ids_path = valid_path + ".ids.question"
data_to_token_ids(dev_path + ".context", x_dis_path, vocab_path)
data_to_token_ids(dev_path + ".question", y_ids_path, vocab_path)

