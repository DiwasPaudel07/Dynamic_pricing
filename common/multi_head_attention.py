# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:15:17 2023

@author: diwaspaudel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads, device):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = input_size // num_heads
        self.input_size = input_size

        self.query_proj = nn.Linear(input_size, input_size).to(device)
        self.key_proj = nn.Linear(input_size, input_size).to(device)
        self.value_proj = nn.Linear(input_size, input_size).to(device)
        self.output_proj = nn.Linear(input_size, input_size).to(device)
        
        init.xavier_uniform_(self.query_proj.weight)
        init.xavier_uniform_(self.key_proj.weight)
        init.xavier_uniform_(self.value_proj.weight)
        init.xavier_uniform_(self.output_proj.weight)


    def forward(self, inputs):
        batch_size = inputs.size(0)
        
        query = self.query_proj(inputs)
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)

        query = query.view(batch_size * self.num_heads, -1, self.head_size)
        key = key.view(batch_size * self.num_heads, -1, self.head_size)
        value = value.view(batch_size * self.num_heads, -1, self.head_size)

        scores = torch.bmm(query, key.transpose(1, 2)) / (self.head_size ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.bmm(attention_weights, value)

        attended_values = attended_values.view(batch_size, -1, self.input_size)
        outputs = self.output_proj(attended_values)

        return outputs