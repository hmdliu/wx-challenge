
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from category_id_map import CATEGORY_ID_LIST


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        bert_output_size = 768
        self.fusion = ConcatDenseSE(args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(p=args.final_dropout),
            nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))
        )
        self.simam = args.simam
        self.co_att = args.co_att
        if args.simam:
            self.refine = SimAM_Block()
        if args.co_att:
            assert args.vlad_hidden_size == bert_output_size
            self.satt1 = MultiHeadAttentionOne(
                n_head=args.co_att_head,
                d_model=bert_output_size,
                d_k=bert_output_size,
                d_v=bert_output_size
            )
            self.satt2 = MultiHeadAttentionOne(
                n_head=args.co_att_head,
                d_model=bert_output_size,
                d_k=bert_output_size,
                d_v=bert_output_size
            )
            self.att1 = MultiHeadAttentionOne(
                n_head=args.co_att_head,
                d_model=bert_output_size,
                d_k=bert_output_size,
                d_v=bert_output_size
            )
            self.att2 = MultiHeadAttentionOne(
                n_head=args.co_att_head,
                d_model=bert_output_size,
                d_k=bert_output_size,
                d_v=bert_output_size
            )
            self.pooler = deepcopy(self.bert.pooler)
            

    def forward(self, inputs, inference=False):
        
        # vision feats
        if self.simam:
            frame_feats = self.refine(inputs['frame_input'].permute(0, 2, 1).unsqueeze(3))
            vision_embedding = self.nextvlad(frame_feats.squeeze(3).permute(0, 2, 1))
        else:
            vision_embedding = self.nextvlad(inputs['frame_input'])
        vision_embedding = self.enhance(vision_embedding)

        # text feats
        # bert_embedding = self.bert(inputs['text_input'], inputs['text_mask'])['pooler_output']
        bert_embedding = self.bert(inputs['text_input'], inputs['text_mask'])['last_hidden_state']

        # co-attention
        if self.co_att:
            vision_embedding = vision_embedding.unsqueeze(1)
            # self-attention
            vision_embedding = self.satt1(vision_embedding, vision_embedding, vision_embedding)
            bert_embedding = self.satt2(bert_embedding, bert_embedding, bert_embedding)
            # cross-attention
            vision_embedding = self.att1(vision_embedding, bert_embedding, bert_embedding)
            bert_embedding = self.att2(bert_embedding, vision_embedding, vision_embedding)
            vision_embedding = vision_embedding.squeeze(1)
            bert_embedding = self.pooler(bert_embedding)

        final_embedding = self.fusion([vision_embedding, bert_embedding])
        prediction = self.classifier(final_embedding)

        if inference:
            return prediction
            # return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs):
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttentionOne(nn.Module):
    """
    Multi-Head Attention module with shared projection
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttentionOne, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qkvs = nn.Linear(d_model, n_head * d_k, bias=False)
        nn.init.normal_(self.w_qkvs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, query_input=False):

        # q: [b, l1, c]; k & v: [b, l2, c] 

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qkvs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_qkvs(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_qkvs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # [(n*b), lq, dk]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # [(n*b), lk, dk]
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # [(n*b), lv, dv]

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # [b, lq, (n*dv)]

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


# SimAM Block (Parameter-free 3D Attention)
class SimAM_Block(torch.nn.Module):
    def __init__(self, lamb=1e-4):
        super().__init__()
        self.act = nn.Sigmoid()
        self.lamb = lamb

    def forward(self, x):
        _, _, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.lamb)) + 0.5
        return x * self.act(y)


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding
