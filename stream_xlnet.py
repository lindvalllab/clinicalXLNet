import numpy as np
import pandas as pd
import torch
from torch.utils import data

from tokenization_xlnet import XLNetTokenizer

# we use the same tokenizer as the base
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# maximum sequence length
max_num_notes = 32
max_seq_len = 512
doc_emb_size = 768
    
class Data_Encoder_FAST(data.Dataset):

    def __init__(self, list_IDs, labels, df):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label        
        y = self.labels[index]
        index = self.list_IDs[index]
        doc_seqs = torch.cat(list(self.df[self.df.HADM_ID == index].DOC_EMB.values), 0)

        xlnet_outputs = torch.zeros(size=(max_num_notes, doc_emb_size), dtype=torch.float)        
        if len(doc_seqs) > max_num_notes:
            xlnet_outputs[:max_num_notes] = doc_seqs[:max_num_notes]
        else:
            xlnet_outputs[:len(doc_seqs)] = doc_seqs

        return xlnet_outputs.cuda(), y
   
    
class Data_Encoder_Seq(data.Dataset):

    def __init__(self, list_IDs, labels, df):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label        
        y = self.labels[index]
        index = self.list_IDs[index]
        seq = self.df.iloc[index].TEXT
        if str(seq) == 'nan':
            seq = 'null text values'
            print (self.df.iloc[index].HADM_ID)
        d_input_ids, d_input_mask, d_segment_ids = preprocess(seq, tokenizer, max_seq_length = max_seq_len)
                
        return d_input_ids, d_segment_ids, d_input_mask, y


def preprocess(datapoint, tokenizer, max_seq_length,
                                 cls_token_at_end=True,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=2,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):    

    
    inputs = tokenizer.encode_plus(
            datapoint,
            None,
            add_special_tokens=True,
            max_length=max_seq_length,
        )
    
    tokens = inputs['input_ids']
    
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[:(max_seq_length - special_tokens_count)]
    # "<SEP> token index"
    tokens = tokens + [4] 
    
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        # "<CLS> token index"
        tokens = tokens + [3] 
        segment_ids = segment_ids + [cls_token_segment_id]
    
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(tokens)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(tokens)
   
    input_ids = tokens + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return np.array(input_ids), np.array(input_mask), np.array(segment_ids)
