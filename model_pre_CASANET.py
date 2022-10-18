# coding:utf-8
import sys

sys.path.append('../')
import torch
import numpy as np
import torch.nn as nn


from CASANET import CASANEncoder
import torch.nn.functional as F



class CASANASDE(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim
        self.args = args
        self.enc = ASDEEncoder(args)
        self.classifier = nn.Linear(in_dim, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        outputs = self.enc(inputs)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        return logits, outputs


class ASDEEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, input, mas, l):
        adj_list = []
        matx_list = []
        adj = np.concatenate(adj_list, axis=0)
        adj = torch.from_numpy(adj).cuda()

        matx = np.concatenate(matx_list, axis=0)
        matx_all = torch.from_numpy(matx).cuda()
        if self.args.model.lower() == "std":
            h = self.encoder(adj=None, inputs=input, lengths=l)
        elif self.args.model.lower() == "gat":
            h = self.encoder(adj=adj, inputs=input, lengths=l)
        elif self.args.model.lower() == "Casanet":
            h = self.encoder(
                adj=adj, matx=matx_all, inputs=input, lengths=l
            )
        else:
            print(
                "Invalid model name {}, it should be (std, GAT, CASANET)".format(
                    self.args.model.lower()
                )
            )
            exit(0)

        graph_out, bs_pool_output, bs_out = h[0], h[1], h[2]
        asp_wn = mas.sum(dim=1).unsqueeze(-1)
        mas = mas.unsqueeze(-1).repeat(1, 1, self.args.out_feat_dim)
        graph_enc_outputs = (graph_out * mas).sum(dim=1) / asp_wn


        if self.args.output_merge.lower() == "none":
            merged_outputs = graph_enc_outputs

        else:
            print('Invalid output_merge type !!!')
            exit()
        cat_outputs = torch.cat([merged_outputs, bs_pool_output], 1)
        return cat_outputs


class LGEncoder(nn.Module): 
    def __init__(self, args, embeddings=None, use_glo=False):
        super(LGEncoder, self).__init__()
        self.args = args
        self.in_drop = nn.Dropout(args.dropout)
        self.dense = nn.Linear(args.hidden_dim, args.out_feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_dim, 2 * args.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * args.hidden_dim, args.hidden_dim)
        )

        if use_glo:
            self.feat_emb, self.out_feat_emb, self.dep_emb = embeddings
            self.Graph_encoder = CASANEncoder(
                num_layers=args.num_layer,
                d_model=args.out_feat_dim,
                heads=4,
                d_ff=args.hidden_dim,
                dep_dim=self.args.dep_dim,
                att_drop=self.args.att_dropout,
                dropout=0.0,
                use_structure=True
            )
        else:
            self.feat_emb, self.out_feat_emb = embeddings  
            self.Graph_encoder = CASANEncoder(
                num_layers=args.num_layer,
                d_model=args.out_feat_dim,
                heads=4,
                d_ff=args.hidden_dim,
                dropout=0.0
            )
        self.e_LGE_embedding1 = nn.Embedding(5, args.hidden_dim)  #num_bond_type = 5
        self.e_LGE_embedding2 = nn.Embedding(3, args.hidden_dim)   #num_bond_direction = 3

        nn.init.xavier_uniform_(self.e_LGE_embedding1.weight.data)
        nn.init.xavier_uniform_(self.e_LGE_embedding2.weight.data)


    def reset_params(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, input):
        x = input.x
        e_index = input.e_index
        e_attr = input.e_attr
        h = CASANEncoder.self.CASAN1(x[:, 0]) + CASANEncoder.self.CASAN2(x[:, 1])
        for layer in range(CASANEncoder.self.num_layers):
            h = self.gnns[layer](h, e_index, e_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, CASANEncoder.self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), CASANEncoder.self.dropout, training=self.training)

        h = self.pool(h, input.batch)
        h = self.feat_lin(h)
        
        h = self.pool(h, input.batch)
        h = self.feat_lin(h)
        logist_h = self.out_lin(h)

        return h, logist_h

if __name__ == "__main__":
    model =LGEncoder()
    print(model)
