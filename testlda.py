import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import os

from gensim import corpora, similarities, models

import gensim

def ReLU(x):
    y = np.maximum(0.0, x) 
    return(y)




def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab



def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def prec_process(revs):
	texts = [ [word for word in rev["text"].split() ] for rev in revs]
	return texts
	


def get_doc_topic(corpus, model): 
    doc_topic = list() 
    for doc in corpus: 
        doc_topic.append(model.__getitem__(doc, eps=0)) 
    return doc_topic 



def get_topic_to_wordids(model): 
    p = list() 
    for topicid in range(model.num_topics): 
        topic = model.state.get_lambda()[topicid] 
        topic = topic / topic.sum() # normalize to probability dist 
        p.append(topic) 
    return p 


def get_topic_to_sentenceword(model, topicswordids, topicsdocument, raw_corpus, dicts): 
    p = list() 
    i = 0 
    for document in raw_corpus:
    	topic_sent = list()
    	for word in document:
            topic_document = topicsdocument[i]
            topic_word = list()
            for topicid in range(model.num_topics): 
                topic_word.append(topic_document[topicid][1] * topicswordids[topicid][dicts.token2id.get(word)])
            topic_wordnorm = [float(topic1)/sum(topic_word) for topic1 in topic_word]
            topic_sent.append(topic_wordnorm)
        p.append(topic_sent)
    	i = i + 1

    return p 


def load_bin_vec(fname, dictionary):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if dictionary.token2id.get(word) > 0:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs


def rand_TE(num_topics=5, k=20):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    W = np.random.uniform(-0.5,0.5,(num_topics,k))    
    return W

def add_unknown_words(word_vecs, dictionary, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in dictionary.itervalues():
            if word not in word_vecs:
                word_vecs[word] = np.random.uniform(-0.25,0.25,k) 

def get_idx_from_sent(sent, topic_weights, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    t = []
    num_topics = len(topic_weights[0])
    pad = filter_h - 1
    zero_weights = np.zeros(num_topics)
    for i in xrange(pad):
        x.append(0)
        t.append(zero_weights)
    words = sent.split()
    for index, word in enumerate(words):
        if word in word_idx_map:
            x.append(word_idx_map[word])
        t.append(topic_weights[index])
    while len(x) < max_l+2*pad:
        x.append(0)
        t.append(zero_weights)
    return x,t

def make_idx_data_cv(revs, lda_weights, word_idx_map, cv, max_l=56, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    train_weight, test_weight = [], []
    for index, rev in enumerate(revs): 
        sent,sent_weight = get_idx_from_sent(rev["text"], lda_weights[index], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        if rev["split"] in cv:            
            test.append(sent)
            test_weight.append(sent_weight)       
        else:  
            train.append(sent)
            train_weight.append(sent_weight)   
    train = np.array(train, dtype="int")
    train_weight = np.array(train, dtype="float32")
    test_weight = np.array(train, dtype="float32")
    vocab_train = np.unique(train)
    test = np.array(test, dtype="int")
    vocab_test = np.unique(test)
    vocab_unseen = np.setdiff1d(vocab_test, vocab_train)
    return [train, test], [train_weight,test_weight], vocab_unseen
  
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

data_folder = ["D:\\ZhaoRui\\KIM\datasets\\rt-polarity.pos","D:\\ZhaoRui\\KIM\\datasets\\rt-polarity.neg"]    
     
revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
print "data loaded"
raw_text = prec_process(revs)
dictionary = corpora.Dictionary(raw_text)

corpus = [dictionary.doc2bow(text) for text in raw_text]

corpora.MmCorpus.serialize('questions.mm', corpus)
mm = corpora.MmCorpus('questions.mm')

num_topics=5

print "LDA model training:"


lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=num_topics, update_every=0, chunksize=19188, passes=20)

lda.save('model')
print "LDA model saved"

Q = get_topic_to_wordids(lda) 

P = get_doc_topic(corpus, lda)



LDAFilter = get_topic_to_sentenceword(lda, Q, P, raw_text, dictionary)
print "LDA weights learned"


w2v = load_bin_vec('D:\\ZhaoRui\\KIM\\datasets\\w2v_file.bin', dictionary)
print "word2vec loaded!"
print "num words already in word2vec: " + str(len(w2v))
add_unknown_words(w2v, dictionary)
W, word_idx_map = get_W(w2v)



datasets, index_unseen = make_idx_data_cv(revs, word_idx_map, cv[i], max_l=max_l,k=300, filter_h=5)









#TTT = conv_sentence(raw_corpus, H, w2v, 3, num_topics)




   
