
from __future__ import print_function
import torch
import torch.nn as nn


class SupCLLoss(nn.Module):

    def __init__(self, temperature=0.1, ctr_size='all',
                 ba_temperature=0.07):
        super(SupCLLoss, self).__init__()
        self.temperature = temperature
        self.ctr_size = ctr_size
        self.ba_temperature = ba_temperature

    def forward(self, in_feat, in_label=None, mask=None):
        device = (torch.device('cuda')
                  if in_feat.is_cuda
                  else torch.device('cpu'))

        in_feat = in_feat.view(in_feat.shape[0], in_feat.shape[1], -1)

        batch_size = in_feat.shape[0]
        if in_label is not None and mask is not None:
            raise ValueError('Cannot define both `in_label` and `mask`')
        elif in_label is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif in_label is not None:
            in_label = in_label.contiguous().view(-1, 1)
            if in_label.shape[0] != batch_size:
                raise ValueError('Num of in_label does not match num of in_feat')
            mask = torch.eq(in_label, in_label.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contra_count = in_feat.shape[1]
        contra_feat = torch.cat(torch.unbind(in_feat, dim=1), dim=0)
        if self.ctr_size == 'one':
            contra_anchor_feat = in_feat[:, 0]
            contra_anchor_num = 1
        elif self.ctr_size == 'all':
            contra_anchor_feat = contra_feat
            contra_anchor_num = contra_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.ctr_size))

        anchor_dot_contrast = torch.div(
            torch.matmul(contra_anchor_feat, contra_feat.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        mask = mask.repeat(contra_anchor_num, contra_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contra_anchor_num).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.ba_temperature) * mean_log_prob_pos
        loss = loss.view(contra_anchor_num, batch_size).mean()

        return loss
