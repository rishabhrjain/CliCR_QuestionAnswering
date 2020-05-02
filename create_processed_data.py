import os
import json
import numpy as np
from pytorch_transformers import BertModel,BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from my_process_json import *
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from tensorflow.keras.preprocessing.sequence import pad_sequences
import copy
from torch.utils import data

#CONSTANTS
DATASET_SIZE = 53914
MAX_LEN = 424
SEED = 520
batch_size = 16



class DataProcessor(data.Dataset):
    
    def __init__(self,filename = 'train', size = None, tokenizer=None):
        data_reader = MyDataReader('/home/jesus/Downloads/NLP_Project/clicr_dataset/'+filename+'1.0.json',bs=size)
        mydata = data_reader.send_batches()
        self.dataset_size = data_reader.get_data_size()
        self.paragraphs = [e['p'] for e in mydata].copy()
        self.paragraph_tags = [e['p_tags'] for e in mydata].copy()
        self.queries = [e['q'] for e in mydata].copy()
        self.query_tags = [e['q_tags'] for e in mydata].copy()

        assert all(len(self.paragraphs) == len(y) for y in [self.paragraph_tags, self.queries, self.query_tags])

        #creating embeddings
        self.tags_vals = ['B-ans','I-ans','O']
        self.tag2idx = {t: i for i, t in enumerate(self.tags_vals)}
        self.tag2idx['[PAD]'] = -100
        self.tag2idx['[SEP]'] = -100
        self.tag2idx['[CLS]'] = -100
        
        if tokenizer == None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            self.tokenizer = tokenizer

    def __getitem__(self, index):
        return tuple(sample[index] for sample in [self.paragraphs, self.paragraph_tags, self.queries, self.query_tags])

    def __len__(self):
        return len(self.paragraphs)
        
        
    def create_tokenizedtexts_and_labels(self):
        split_size = 200
        p_tags = copy.deepcopy(self.paragraph_tags)
        q_tags = copy.deepcopy(self.query_tags)
        paragraphs = copy.deepcopy(self.paragraphs)
        queries = copy.deepcopy(self.queries)
        tokenized_texts = []
        labels = []
        for i,paragraph in enumerate(paragraphs):
            texts = []
            paragraph_tag = copy.deepcopy(p_tags[i])
            #number of splits for current paragraph
            n_splits = int(len(paragraph.split())/split_size)
            #tokenize split paragraph text appended by full query at the end with <sep> token between paragraph and query
            queries[i] = queries[i].replace("@placeholder",'[MASK]')
            texts += [s for s in ['[CLS] ' + ' '.join(paragraph.split()[(split_size*split):(split_size*(split+1))]) + ' [SEP] ' + queries[i] + ' [SEP]' for split in range(n_splits)]]
            tokenized_texts += [self.tokenizer.tokenize(s) for s in ['[CLS] ' + ' '.join(paragraph.split()[(split_size*split):(split_size*(split+1))]) + ' [SEP] ' + queries[i] + ' [SEP]' for split in range(n_splits)]]
            #tokenize remainder of splits
            if int(len(paragraph.split()) % split_size) > 0:
                texts += ['[CLS] ' + ' '.join(paragraph.split()[(n_splits*split_size):]) + ' [SEP] ' + queries[i] + ' [SEP]']
                tokenized_texts += [self.tokenizer.tokenize('[CLS] ' + ' '.join(paragraph.split()[(n_splits*split_size):]) + ' [SEP] ' + queries[i] + ' [SEP]')]

            #create labels to be used for tagging from the text
            for split_index, sent in enumerate(texts):
                l = []
                for word_index,word in enumerate(sent.split()):
                    if '[SEP]' in word:
                        l += ['[SEP]']
                        break
                    if '[CLS]' in word:
                        l += ['[CLS]']
                        continue
                    if not paragraph_tag:
                        print(sent,'\n')
                        print(word)
                        print(split_index)
                    lab = paragraph_tag.pop(0)
                    word_list = self.tokenizer.tokenize(word)
                    for w in word_list:
                        l += [lab]
                query_tags = copy.deepcopy(q_tags[i])
                for word in queries[i].split():
                    lab = query_tags.pop(0)
                    word_list = self.tokenizer.tokenize(word)
                    for w in word_list:
                        l += [lab]
                l += ['[SEP]']
                labels += [l]
        self.labels = labels
        self.tokenized_texts = tokenized_texts
    
    
    
    def create_input_ids(self):
        self.input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in self.tokenized_texts],maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    
    
    def create_token_type_ids(self):
        token_type_ids = []
        for ipid in self.input_ids:
            type_id = 0
            token_type_id = []
            for myid in ipid:
                token_type_id.append(type_id)
                if myid == 102: 
                    if type_id%2==0:
                        type_id+= 1
                    else:
                        type_id=0
            token_type_ids.append(token_type_id)
        self.token_type_ids = token_type_ids
    

    
    def create_tags(self):
        self.tags = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in self.labels],
                     maxlen=MAX_LEN, value=self.tag2idx["[PAD]"], padding="post",
                     dtype="long", truncating="post")
        
    
    def create_attention_masks(self):
        self.attention_masks = [[float(i>0) for i in ii] for ii in self.input_ids]

        
    def get_tags_vals(self):
        return self.tags_vals
        
        
    def get_processed_data(self):
        self.create_tokenizedtexts_and_labels()
        self.create_input_ids()
        self.create_token_type_ids()
        self.create_tags()
        self.create_attention_masks()
        
        return self.input_ids, self.attention_masks, self.tags, self.token_type_ids, self.tokenizer
