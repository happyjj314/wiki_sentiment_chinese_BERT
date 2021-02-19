import numpy as np
import random
import torch
import tqdm
import json
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(self,corpus_path,word2idx_path,seq_len,hidden_dim):
        self.hidden_dim = hidden_dim
        self.corpus_path = corpus_path
        self.word2idx_path = word2idx_path
        self.seq_len = seq_len
        self.pad = 0
        self.unk = 1
        self.cls = 2
        self.sep = 3
        self.mask = 4
        self.num = 5

        # load dic
        with open(self.word2idx_path,'r',encoding='utf-8') as f:
            self.word2idx = json.load(f)

        # load corpus
        with open(self.corpus_path,'r',encoding='utf-8') as f:
            self.lines = [eval(line) for line in tqdm.tqdm(f,desc="开始加载语料")]
            self.corpus_length = len(self.lines)


    def __len__(self):
        return self.corpus_length

    def __getitem__(self, item):
        t1,t2,is_next = self.random_sent(item)

        t1_finished_mask,t1_mask_token = self.random_char(t1)
        t2_finished_mask,t2_mask_token = self.random_char(t2)

        # 句子头尾加 CLS 和SEP
        t1 = [self.cls] + t1_finished_mask + [self.sep]
        t2 = t2_finished_mask + [self.sep]

        # 给t1_mask_token 也加上 cls 不过加的是pad 为了保证长度一样
        t1_mask_token = [self.pad] + t1_mask_token + [self.pad]
        t2_mask_token = t2_mask_token + [self.pad]

        # seq_len 截断作用
        segment_label = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]

        bert_input = (t1+t2)[:self.seq_len]
        bert_mask_label = (t1_mask_token+ t2_mask_token)[:self.seq_len]

        # return bert_input,bert_mask_label,segment_label,is_next

        output = {"bert_input": torch.tensor(bert_input),
                  "bert_label": torch.tensor(bert_mask_label),
                  "segment_label": torch.tensor(segment_label),
                  "is_next": torch.tensor([is_next])}
        return output

    def random_sent(self,index):
        t1,t2 = self.lines[index]['text1'],self.lines[index]['text2']
        if random.random() >0.5:
            return t1,t2,1 # 上下句
        else:
            # t2随机给一个句子
            return t1,self.lines[random.randrange(len(self.lines))]['text2'],0
    def random_char(self,sentence):

        char_tokens = [self.word2idx.get(char,self.unk) for char in list(sentence)]
        output_label = []
        for i,token in enumerate(char_tokens):
            prob =random.random()
            if prob < 0.3:
                prob /= 0.3
                # 被mask前这里是什么
                output_label.append(token)
                if prob < 0.8:
                    # 把这个位置mask用
                    char_tokens[i] = self.mask
                elif prob <0.9:
                    # 随机mask
                    char_tokens[i] = random.randrange(len(self.word2idx))
            else:
                # append 0表示这个位置是没有mask的
                output_label.append(0)

        return char_tokens,output_label

