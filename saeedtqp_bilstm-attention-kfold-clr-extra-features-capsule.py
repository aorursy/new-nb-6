import time

import random

import pandas as pd

import numpy as np

import gc

import re

import torch

from torchtext import data

import spacy

from tqdm import tqdm_notebook, tnrange

from tqdm.auto import tqdm



tqdm.pandas(desc='Progress')

from collections import Counter

from textblob import TextBlob

from nltk import word_tokenize



import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.autograd import Variable

from torchtext.data import Example

from sklearn.metrics import f1_score

import torchtext

import os 



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# cross validation and metrics

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score

from torch.optim.optimizer import Optimizer

from unidecode import unidecode
# Basic Parameters



embed_size = 300  # how big is each word vector

max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 70 # max number of words in a question to use

batch_size = 512 # how many samples to process at once

n_epochs = 5 # how many times to iterate over all samples

n_splits = 5 # Number of K-fold splits



SEED = 1029
def seed_everything(seed=1029):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything()
# Code for loading Embeddings (GLOVE)



def load_glove(word_index):

    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/'

    

    def get_coefs(word,*arr):return word , np.asarray(arr,dtype='float32')[:300]

    

    embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(EMBEDDING_FILE))

    

    all_embs = np.stack(embeddings_index.values())

    emb_mean , emb_std = -0.005838499,0.48782197

    embed_size = all_embs.shape[1]

    

    # word_index = tokenizer.word_index

    

    nb_words = min(max_features,len(word_index))

    embedding_matrix = np.random_normal(emb_mean , emb_std , (nb_words,embed_size))

    for word , i in word_index.items():

        if i >= max_features:continue

        embedding_vector = embeddings_index.get(word)

        

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

            

    return embedding_matrix
# Load FASTEXT



def load_fasttext(word_index):    

    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector



    return embedding_matrix
def load_para(word_index):

    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = -0.0053247833,0.49346462

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    

    return embedding_matrix
# Load Processed Training DATA FROM Disk



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



df = pd.concat([df_train , df_test],sort = True)
def build_vocab(texts):

    sentences = texts.apply(lambda x: x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab

vocab = build_vocab(df['question_text'])
sin = len(df_train[df_train['target'] == 0])

insin = len(df_train[df_train['target'] == 1])

persin = (sin / (sin/insin)) * 100

perinsin = (insin / (sin + insin)) * 100



print('# sincere questions : {:,}({:.2f}%) and # Insincers Questions : {:,}({:.2f}%)'.format(sin,persin ,insin , perinsin))



print('# test samples:{:,}({:.3f} % of train samples)'.format(len(df_test),len(df_test)/len(df_train)))

# Normalization



def build_vocab(texts):

    sentences = texts.apply(lambda x:x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
def known_contractions(embed):

    known = []

    

    for contract in contraction_mapping:

        if contract in embed:

            known.append(contract)

    return known



# ---



def clean_contractions(text,mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s,"'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(' ')])

    

    return text



# ---



def correct_spelling(x , dic):

    for word in dic.keys():

        x  = x.replace(word , dic[word])

    return x



# ---



def unknown_punct(embed , punct):

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown



# ---



def clean_numbers(x):

    x = re.sub('[0-9]{5,}','#####',x)

    x = re.sub('[0-9]{4}','###',x)

    x = re.sub('[0-9]{3}','##', x)

    x = re.sub('[0-9]{2}', '##',x)

    return x



# ----



def clean_special_chars(text,punct , mapping):

    for p in mapping:

        text = text.replace(p,mapping[p])

        

    for p in punct:

        text = text.replace(p,f'{p} ')

        specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}

        

        for s in specials:

            text = text.replace(s,specials[s])

            

    return text



# ----



def add_lower(embedding , vocab):

    

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:

            embedding[word.lower()] = embedding[word]

            count += 1

            

    print(f'added {count} words to embedding')

    

        
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):

    x = str(X)

    for punct in puncts:

        x = x.replace(punct , f'{punct} ')

    return x
mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict , mispell_re



misspellings , mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):

    def replace(match):

        return misspellings[match.group(0)]

    return misspellings_re.sub(replace , text)