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
    		topic_sent.append(topic_word)
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

def add_unknown_words(word_vecs, dictionary, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in dictionary.itervalues():
            if word not in word_vecs:
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


cPickle.dump([revs, raw_text, LDAFilter, w2v, dictionary, num_topics], open("mr.p", "wb"))
print "dataset created!"











#TTT = conv_sentence(raw_corpus, H, w2v, 3, num_topics)




   
