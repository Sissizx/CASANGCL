""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):

    def __init__(self, n_head, in_feat_dim, out_feat_dim=30, dropout=0.1, struct_i_j=False, alpha=1.0, beta=1.0):
        assert in_feat_dim % n_head == 0
        self.per_head_dim = in_feat_dim // n_head
        self.in_feat_dim = in_feat_dim

        super(MultiHeadedAttention, self).__init__()
        self.n_head = n_head

        self.linear_k = nn.Linear(in_feat_dim, n_head * self.per_head_dim)
        self.linear_v = nn.Linear(in_feat_dim, n_head * self.per_head_dim)
        self.linear_q = nn.Linear(in_feat_dim, n_head * self.per_head_dim)
        if struct_i_j:
            self.struct_k = nn.Linear(out_feat_dim, self.per_head_dim)
            self.struct_v = nn.Linear(out_feat_dim, self.per_head_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(in_feat_dim, in_feat_dim)
        self.alpha = alpha
        self.beta = beta

    def forward(self, key, value, query, structure=None, mask=None, k_pad_mask=None, l_cach=None,type=None,):

        bs = key.size(0)
        per_head_dim = self.per_head_dim
        n_head = self.n_head
        key_len = key.size(1)
        query_len = query.size(1)

        def view_z(x):
            """  projection """
            return x.view(bs, -1, n_head, per_head_dim).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, n_head * per_head_dim)

        #  Project key, value, and query.
        if l_cach is not None:
            if type == "self":
                query, key, value = (
                    self.linear_q(query),
                    self.linear_k(query),
                    self.linear_v(query),
                )
                if structure is not None:
                    structure_k, structure_v = (
                        self.struct_k(structure),
                        self.struct_v(structure),
                    )
                else:
                    structure_k = None
                    structure_v = None

                key = view_z(key)
                value = view_z(value)

                if l_cach is not None:
                    device = key.device
                    if l_cach["self_keys"] is not None:
                        key = torch.cat((l_cach["self_keys"].to(device), key), dim=2)
                    if l_cach["self_values"] is not None:
                        value = torch.cat((l_cach["self_values"].to(device), value), dim=2)
                    l_cach["self_keys"] = key
                    l_cach["self_values"] = value

            elif type == "context":
                query = self.linear_q(query)
                if l_cach is not None:
                    if l_cach["memory_keys"] is None:
                        key, value = self.linear_k(key), self.linear_v(value)
                        key = view_z(key)
                        value = view_z(value)
                    else:
                        key, value = l_cach["memory_keys"], l_cach["memory_values"]
                    l_cach["memory_keys"] = key
                    l_cach["memory_values"] = value
                else:
                    key, value = self.linear_k(key), self.linear_v(value)
                    key = view_z(key)
                    value = view_z(value)
        else:
            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)
            if structure is not None:
                structure_k, structure_v = (
                    self.struct_k(structure),
                    self.struct_v(structure),
                )
            else:
                structure_k = None
                structure_v = None

            key = view_z(key)
            value = view_z(value)

        query = view_z(query)


        key_len = key.size(2)
        query_len = query.size(2)

        # Calculate and scale scores.
        query = query / math.sqrt(per_head_dim)
        scores = torch.matmul(query, key.transpose(2, 3))


        if structure_k is not None:
            q = query.transpose(1, 2)

            calcu_k = torch.matmul(
                q, structure_k.transpose(2, 3)
            )
            calcu_k = calcu_k.transpose(1, 2)

            scores = scores + self.alpha * calcu_k


        if k_pad_mask is not None:
            k_pad_mask = k_pad_mask.unsqueeze(1).unsqueeze(2)

            scores = scores.masked_fill(k_pad_mask, -1e18)


        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e18)


        atten_weight = self.softmax(scores)
        attn = self.dropout(atten_weight)
        context = torch.matmul(attn, value)

        if structure_v is not None:
            drop_attn_v = attn.transpose(1, 2)
            context_v = torch.matmul(drop_attn_v, structure_v)
            context_v = context_v.transpose(1, 2)
            context = context + self.beta * context_v

        context = unshape(context)
        output = self.final_linear(context)

        fin_attn = atten_weight.view(bs, n_head, query_len, key_len)[:, 0, :, :].contiguous()

        return output, fin_attn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

