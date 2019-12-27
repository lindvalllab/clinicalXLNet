from __future__ import print_function
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import collections
import math
import copy
torch.manual_seed(1)
np.random.seed(1)

from modeling_xlnet import XLNetModel, XLNetPreTrainedModel
from modeling_utils import PreTrainedModel, prune_linear_layer, SequenceSummary, PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits
               
class clinical_xlnet_lstm_FAST(nn.Sequential):
    
    def __init__(self):
            super(clinical_xlnet_lstm_FAST, self).__init__()
            
            self.intermediate_size = 1536
            self.num_attention_heads = 12
            self.attention_probs_dropout_prob = 0.1
            self.hidden_dropout_prob = 0.1
            self.hidden_size_encoder = 768
            self.n_layers = 2
            self.hidden_size_xlnet = 768
            
            self.encoder = nn.LSTM(self.hidden_size_encoder,self.hidden_size_encoder, 2, bidirectional = True)    
            self.decoder = nn.Sequential(
                nn.Dropout(p=self.hidden_dropout_prob),
                nn.Linear(self.hidden_size_encoder*2, 32),
                nn.ReLU(True),
                #output layer
                nn.Linear(32, 1)
            )
            
    def forward(self, xlnet_outputs):
           
            self.encoder.flatten_parameters()
            output, (_, _) = self.encoder(xlnet_outputs.permute(1,0,2))
    
            last_layer = output[-1]
            score = self.decoder(last_layer)
            
            return score
               
        
class clinical_xlnet_seq(XLNetPreTrainedModel):
    
    def __init__(self, config):
            super(clinical_xlnet_seq, self).__init__(config)
           
            self.hidden_size_xlnet = 768
            
            self.transformer = XLNetModel(config)
            self.sequence_summary = SequenceSummary(config)
         
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden_size_xlnet, 32),
                nn.ReLU(True),
                #output layer
                nn.Linear(32, 1)
            )
            
    def forward(self, input_ids, seg_ids, masks):
            output = self.sequence_summary(self.transformer(input_ids, token_type_ids = seg_ids, attention_mask = masks)[0])
            
            score = self.decoder(output)
            
            return score, output        
        