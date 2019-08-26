#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@paper: Improving Web Image Search using Contextual Information
@model: CARM & NARM
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from config import *
from torch.nn import LayerNorm
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(1)


class CARM(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, pretrained_image_embedding, pretrained_word_embedding,
                 n_layers = 1):
        super(CARM, self).__init__()
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.image_gru = nn.GRU(200, 200)
        self.ln1 = nn.Linear(200, 200)
        self.ln2 = nn.Linear(200, 200)
        self.img_embedding = nn.Embedding(len(pretrained_image_embedding), hidden_size)
        self.img_embedding.weight = nn.Parameter(torch.from_numpy(pretrained_image_embedding))
        self.word_embedding = nn.Embedding(len(pretrained_word_embedding), 200)
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_embedding))
        self.cos = nn.CosineSimilarity(dim=1, eps=MINF)
        self.dropout = nn.Dropout(p = DROP_OUT_PROB)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.query_ln = nn.Linear(200, 200)
        self.image_ln = nn.Linear(512, 200)
        self.trade_off_parameter = 0.5
        self.feature_to_feature = nn.Linear(6, 6)
        self.feature_to_para = nn.Linear(6, 1)

    def forward(self, source_query, source_image, source_image_feature, source_image_cnt, target_query, target_image, target_score):
        now_batch_size = source_query.shape[0]
        # Query lookup
        source_query = source_query.view(-1,5)
        source_query = self.word_embedding(source_query).view(now_batch_size,5,5,-1)
        source_query = torch.tanh(self.query_ln(source_query))
        source_query = torch.mean(source_query, 2)
        target_query = self.word_embedding(target_query)
        target_query = torch.tanh(self.query_ln(target_query))
        target_query = torch.mean(target_query, 1)
        query_att = torch.bmm(source_query, target_query.view(now_batch_size,200,1)).view(now_batch_size,5)
        query_att = self.softmax(query_att).view(now_batch_size, 1, 5)
        # Image lookup
        source_image = self.img_embedding(source_image)
        source_image = torch.tanh(self.image_ln(source_image))
        source_image = torch.sum(source_image, 2)
        # Gru
        image_hidden = self.inithidden(now_batch_size, 200)
        source_image = source_image.transpose(0, 1)
        gru_output, hidden = self.image_gru(source_image, image_hidden)
        gru_output = gru_output.transpose(0, 1)
        # Attention layer
        output = torch.bmm(query_att, gru_output).view(now_batch_size, 200)
        source_output = target_query + output

        target_image = self.img_embedding(target_image)
        target_image = torch.tanh(self.image_ln(target_image))
        target_output = target_image

        #Score computation
        score_0 = self.cos(source_output, target_output[:,0,:])
        score_1 = self.cos(source_output, target_output[:,1,:])

        return score_0, score_1


    def inithidden(self, real_batch_size, hidden_size):
        result = Variable(torch.zeros(1, real_batch_size, hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class NARM(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, pretrained_image_embedding, pretrained_word_embedding,
                 n_layers = 1):
        super(NARM, self).__init__()
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.image_gru = nn.GRU(200, 200)
        self.ln1 = nn.Linear(200, 200)
        self.ln2 = nn.Linear(200, 200)
        self.img_embedding = nn.Embedding(len(pretrained_image_embedding), hidden_size)
        self.img_embedding.weight = nn.Parameter(torch.from_numpy(pretrained_image_embedding))
        self.word_embedding = nn.Embedding(len(pretrained_word_embedding), 200)
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_embedding))
        self.cos = nn.CosineSimilarity(dim=1, eps=MINF)
        self.dropout = nn.Dropout(p = DROP_OUT_PROB)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.query_ln = nn.Linear(200, 200)
        self.image_ln = nn.Linear(512, 200)
        self.A_1 = nn.Parameter(torch.randn(200, 200)) #q(h_t, h_j)
        self.A_2 = nn.Parameter(torch.randn(200, 200)) #q(h_t, h_j)
        self.V = nn.Parameter(torch.randn(1, 200)) #q(h_t, h_j)
        self.B = nn.Parameter(torch.randn(200, 400))
        self.layer_norm_ct = LayerNorm(400, eps=1e-12)
        self.layer_norm_ti = LayerNorm(200, eps=1e-12)
        self.q_ln_A_1 = nn.Linear(200, 200, bias=False)
        self.q_ln_A_2 = nn.Linear(200, 200, bias=False)
        self.q_ln_v = nn.Linear(200, 1, bias=False)



    def forward(self, source_query, source_image, source_image_feature, source_image_cnt, target_query, target_image, target_score):
        now_batch_size = source_query.shape[0]
        source_image = self.img_embedding(source_image)
        source_image = self.relu(self.image_ln(source_image))
        source_image = source_image.view(now_batch_size, 25, -1)
        image_hidden = self.inithidden(now_batch_size, 200)
        source_image = source_image.transpose(0, 1)

        gru_output, hidden = self.image_gru(source_image, image_hidden)
        gru_output = gru_output.transpose(0, 1) #h_t_1
        hidden = hidden[0, :, :] #h_t_g
        h_t = gru_output[:, 24, :]
        c_t_g = h_t

        #q(h_t, h_j)
        h_t = h_t.view(now_batch_size, 1, -1).expand(-1, 25, -1)
        alpha = self.sigmoid(self.q_ln_A_1(h_t) + self.q_ln_A_2(gru_output))
        alpha = self.q_ln_v(alpha)
        alpha = self.softmax(alpha).view(now_batch_size, 1, 25)
        c_t_1 = torch.bmm(alpha, gru_output)[:,0,:]
        c_t = self.layer_norm_ct(torch.cat((c_t_1, c_t_g), 1)).view(now_batch_size, 400, 1)

        target_image = self.img_embedding(target_image)
        target_image = self.relu(self.image_ln(target_image))
        target_output = self.layer_norm_ti(target_image)

        B = self.B.view(1,200,400).expand(now_batch_size, -1, -1)
        image_0 = target_output[:,0,:].view(now_batch_size, 1, 200)
        image_1 = target_output[:,1,:].view(now_batch_size, 1, 200)
        score_0 = torch.tanh(torch.bmm(torch.bmm(image_0, B) /200**0.5, c_t)/20)[:, 0, :]
        score_1 = torch.tanh(torch.bmm(torch.bmm(image_1, B) /200**0.5, c_t)/20)[:, 0, :]

        return score_0, score_1


    def inithidden(self, real_batch_size, hidden_size):
        result = Variable(torch.zeros(1, real_batch_size, hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

if __name__ == "__main__":
    pass