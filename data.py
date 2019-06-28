#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:43:29 2019

@author: xpwang
"""
from __future__ import print_function
#from hyperparams import Hyperparams as hp
from pypinyin import pinyin, lazy_pinyin, Style# pip install pypinyin
import codecs
import regex# pip install regex
import pickle
import numpy as np
from hyperparams import Hyperparams as hp
import re
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
# -----------------------------------------------------------------------------------------------------
'''&usage:     generate dataset for training into data/zh.tsv, main function is build_corpus(), style: [key pinyin hanzi]'''
# -----------------------------------------------------------------------------------------------------
def align(sent):
    pnyn = pinyin(sent,style=Style.TONE3) # convert hanzi to pinyin
    hanzis = sent
    pnyns = ''
    for i in pnyn:
        i = "".join(i)
        pnyns = pnyns + i
    hanzis = "".join(hanzis)
    return pnyns, hanzis

def clean(text):
    if regex.search("[A-Za-z0-9]", text) is not None: # For simplicity, roman alphanumeric characters are removed.
        return ""
    text = regex.sub(u"[^ \p{Han}。，！？]", "", text)
    return text

def build_corpus():    
    with codecs.open("data/zh.tsv", 'w', 'utf-8') as fout:
        with codecs.open("data/lable.txt", 'r', 'utf-8') as fin:
            i = 1
            while 1:
                line = fin.readline()
                if not line: 
                    break                
                try:
                    line = line.strip('\n')
                    idx = line.split(' ', 1)[0]
                    sent = line.split(' ', 1)[1]
                    # remove symbols that is not hanzi
                    #print(sent)
                    sent = clean(sent)
                    #print(sent)
                    if len(sent) > 0:
                        pnyns, hanzis = align(sent)
                        fout.write(u"{}\t{}\t{}\n".format(idx, pnyns, hanzis))
                except:
                    continue # it's okay as we have a pretty big corpus!                
                if i % 10000 == 0: print(i, )
                i += 1
# -----------------------------------------------------------------------------------------------------
'''&usage:     generate dict for training, including [pinyin:index] and [hanzi:index] mapping'''
# -----------------------------------------------------------------------------------------------------
#   构造字典, 使用的是基于语料库中的字构造字典，有的人可能会先分词，基于词构造。
#   不使用基于词是现在就算是最好的分词都会有一些误分词问题，而且基于字还可以在一定程度上缓解OOV的问题
def build_vocab():
    from collections import Counter
    from itertools import chain
    pnyn_sents = [(line.split('\t')[1]).split(' ') for line in codecs.open('data/zh.tsv', 'r', 'utf-8').read().splitlines()]
    # make dict based on stat of item frequency
    # generate dict for mapping pinyin to index and save to data/vocab.pkl
    pnyn2cnt = Counter(chain.from_iterable(pnyn_sents))
    pnyns = [pnyn for pnyn, cnt in pnyn2cnt.items() if cnt > 5] # remove long-tail characters
    pnyns = pnyns[0:-1]
    pnyns = ["E", "U", "_" ] + pnyns # 0: empty, 1: unknown, 2: blank
    pnyn2idx = {pnyn:idx for idx, pnyn in enumerate(pnyns)}
    idx2pnyn = {idx:pnyn for idx, pnyn in enumerate(pnyns)}
    
    # generate dict for mapping hanzi to index and save to data/vocab.pkl中
    hanzi_sents = [(line.split('\t')[2]).split(' ') for line in codecs.open('data/zh.tsv', 'r', 'utf-8').read().splitlines()]
    
    hanzi2cnt = Counter(chain.from_iterable(hanzi_sents))
    hanzis = [hanzi for hanzi, cnt in hanzi2cnt.items() if cnt > 5] # remove long-tail characters
    hanzis = hanzis[0:-1]
    hanzis = ["E", "U", "_" ] + hanzis # 0: empty, 1: unknown, 2: blank
    hanzi2idx = {hanzi:idx for idx, hanzi in enumerate(hanzis)}
    idx2hanzi = {idx:hanzi for idx, hanzi in enumerate(hanzis)}

    pickle.dump((pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi), open('data/vocab.pkl', 'wb'), 0)
    return pnyn2idx, hanzi2idx, idx2pnyn, idx2hanzi
def load_vocab():
    import pickle
    return pickle.load(open('data/vocab.pkl', 'rb'))

class HanziDataset(Dataset):
    def __init__(self, tsv_file='data/zh.tsv'):
        """
        Args:
            tsv_file (string): Path to the tsv file with [pinyin, hanzi] annotations
        """
        self.pnyn2idx, self.idx2pnyn, self.hanzi2idx, self.idx2hanzi = load_vocab()
        self.pyhanzifile = open('data/zh.tsv','r')
        self.pyhanziset  = self.pyhanzifile.read()
        self.pyhanzi_lines = self.pyhanziset.split('\n')
        self.pyhanzi_lines.pop(154988)
        self.DataNum     = self.__len__()
        
    def __len__(self):
        return len(self.pyhanzi_lines)
    
    def __getitem__(self, index):
        #index = random.randint(0, self.DataNum)
        txt_s     = self.pyhanzi_lines[index].split('\t')
        #print(txt_s)
        pnyn_vec  = txt_s[1].split()
        hanzi_vec = txt_s[2].split()
        pnyn_ids  = [self.pnyn2idx.get(pnyn,1) for pnyn in pnyn_vec]
        hanzi_ids = [self.hanzi2idx.get(hanzi,1) for hanzi in hanzi_vec]
        return pnyn_ids, hanzi_ids
    
def _collate_fn(batch):
    minibatch_size = len(batch)
    #pys, hzs = [], []
    inputs = torch.zeros(minibatch_size, hp.maxlen)
    targets = torch.zeros(minibatch_size, hp.maxlen)
    for x in range(minibatch_size):
        sample = batch[x]
        py_ids = sample[0]
        hz_ids = sample[1]
        
        seq_length = len(py_ids)
        inputs[x].narrow(0,0,seq_length).copy_(torch.IntTensor(py_ids))
        
        seq_length = len(hz_ids)
        targets[x].narrow(0,0,seq_length).copy_(torch.IntTensor(hz_ids))
        
        #pys.append(np.array(py_ids, np.int32))
        #hzs.append(np.array(hz_ids, np.int32))
        #pys.extend(torch.IntTensor(py_ids))
        #hzs.extend(torch.IntTensor(hz_ids))
    #pys = torch.IntTensor(pys)
    #hzs = torch.IntTensor(hzs)
    return inputs, targets


class PYHZDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(PYHZDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn