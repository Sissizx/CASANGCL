from model_sublayer import PositionwiseFeedForward, MultiHeadedAttention
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from model_pre_CASANET import LGEncoder

class CASALayer(nn.Module):
    def __init__(
        self, d_model, n_head, d_ff, dropout, att_drop=0.1, struct_i_j=True, dep_dim=30, alpha=1.0, beta=1.0, pool='mean'
    ):
        super(CASALayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            n_head, d_model, dropout=att_drop, struct_i_j=struct_i_j, structure_dim=dep_dim, alpha=alpha, beta=beta
        )

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  
        self.dropout = nn.Dropout(dropout)

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

    def forward(self, x,e_feat,e_attr):

        e_feat = add_self_loops(e_feat, num_nodes=x.size(0))[0]
        Wh = torch.mm(x, self.W)

        e = self.input_attn(Wh) 
        zero_vec = -9e15 * torch.ones_like(e)

        attn = torch.where(e_feat > 0, e, zero_vec) 
        attn = F.softmax(attn, dim=1)  
        attn = F.dropout(attn, self.dropout, training=self.training)
        h_prime = torch.matmul(attn, Wh)  


        self_attr = torch.zeros(x.size(0), 2)
        self_attr[:, 0] = 4
        self_attr = self_attr.to(e_attr.device).to(e_attr.dtype)
        e_attr = torch.cat((e_attr, self_attr), dim=0)

        e_embeddings = LGEncoder.self.e_LGE_embedding1(e_attr[:, 0]) + \
                          LGEncoder.self.e_LGE_embedding1(e_attr[:, 1])


        out = self.propagate(e_feat, x=x, e_attr=e_embeddings, size=None)
        return out

    def message(self, x_j, e_attr): 
        return x_j + e_attr

    def message_and_aggregate(self, adj_t, x):  
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr) 

        return self.feed_forward(out)


class CASANEncoder(nn.Module): 

    def __init__(self,num_layers,d_model, n_head,d_ff,dropout,att_drop=0.1,struct_i_j=True,dep_dim=30,alpha=1.0,beta=1.0, ):
        super(CASANEncoder, self).__init__()

        self.num_layers = num_layers
        self.d_model=d_model
        self.dropout=dropout
        self.dep_dim=dep_dim
        self.n_head=n_head
        self.att_drop=att_drop
        self.encoder = nn.ModuleList(
            [
                CASALayer(
                    d_model,
                    n_head,
                    d_ff,
                    dropout,
                    att_drop,
                    att_drop=att_drop,
                    struct_i_j=struct_i_j,
                    dep_dim=dep_dim,
                    alpha=alpha,
                    beta=beta,
                    pool='mean'
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.CASAN1 = nn.Embedding(119, d_model) #num_atom_type = 119
        self.CASAN2 = nn.Embedding(3, d_model)  #num_chirality_tag = 3


        nn.init.xavier_uniform_(self.CASAN1.weight.data)
        nn.init.xavier_uniform_(self.CASAN2.weight.data)

        self.gnns = nn.ModuleList()
        for layer in range(num_layers):
            self.gnns.append(CASALayer(d_model, d_ff, dropout, alpha, concat=True))

        self.batch_norms = nn.ModuleList()
        for layer in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(d_model))


    def _check_args(self, src, lengths=None):
        _, n_batch = src.size()
        if lengths is not None:
            (n_batch_,) = lengths.size()
            # aeq(n_batch, n_batch_)

    def forward(self, ctx):
        x = ctx.x
        e_feat = ctx.e_feat
        e_attr = ctx.e_attr

        m = self.CASAN1(x[:,0]) + self.CASAN2(x[:,1])

        for layer in range(self.num_layers):
            m = self.gnns[layer](m, e_feat, e_attr)
            m = self.batch_norms[layer](m)
            if layer == self.num_layers - 1:
                m = F.dropout(m, self.att_drop, training=self.training)
            else:
                m = F.dropout(F.relu(m), self.att_drop, training=self.training)

        m = self.pool(m, ctx.batch)
        m = self.feat_lin(m)
        out = self.out_lin(m)

        return m, out

if __name__ == "__main__":
    model = CASANEncoder()
    print(model)
