import torch
import numpy as np


class NT_Xent(torch.nn.Module):

    def __init__(self, device, bs, temper, ctx):
        super(NT_Xent, self).__init__()
        self.bs = bs
        self.temper = temper
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples = self.mask_correlated_samples().type(torch.bool)
        self.similarity_function = self.similarity_calculation(ctx)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def similarity_calculation(self, ctx):
        if ctx:
            self.cos_calculate = torch.nn.CosineSimilarity(dim=-1)
            return self.cosine_simililarity_calculation
        else:
            return self.dot_simililarity_calculation


    @staticmethod
    def dot_simililarity_calculation(x, y):
        m = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return m

    def cosine_simililarity_calculation(self, x, y):
        m = self.cos_calculate(x.unsqueeze(1), y.unsqueeze(0))
        return m

    def mask_correlated_samples(self):
        N = 2 * self.bs * self.bs
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.bs * self.bs):
            mask[i, self.bs + i] = 0
            mask[self.bs + i, i] = 0
        return mask.to(self.device)


    def forward(self, z_i, z_j):
        r = torch.cat([z_j, z_i], dim=0)

        similarity_matrix = self.similarity_function(r, r)

        sim_i_j = torch.diag(similarity_matrix, self.bs)
        sim_j_i = torch.diag(similarity_matrix, -self.bs)
        pos_samples = torch.cat([sim_i_j, sim_j_i]).view(2 * self.bs, 1)

        neg_samples = similarity_matrix[self.mask_samples].view(2 * self.bs, -1)

        logits = torch.cat((pos_samples, neg_samples), dim=1)
        logits /= self.temper
        labels = torch.zeros(2 * self.bs).to(self.device).long()
        loss = self.criterion(logits, labels)
        loss = loss / (2 * self.bs)

        return loss
