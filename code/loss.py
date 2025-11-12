import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from collections import defaultdict

    

device = torch.device('cuda:3') 
class GPaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=2048, num_classes=7, smooth=0.0):
        super(GPaCoLoss, self).__init__()

        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.num_classes = num_classes
        self.smooth = smooth
        self.weight = None

    def cal_weight_for_classes(self, cls_num_list):
        # Compute class weights (for handling class imbalance)
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()  
        self.weight = self.weight.to(device)

    def forward(self, features, labels=None, sup_logits=None):
        # Weight coefficient for prototype-to-sample contrast
        batch_size = features.shape[0] - self.K
       
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        if self.weight is not None:
            anchor_dot_contrast = torch.cat(((sup_logits + torch.log(self.weight + 1e-9)) / self.supt, anchor_dot_contrast), dim=1)
        else:
            anchor_dot_contrast = torch.cat((sup_logits / self.supt, anchor_dot_contrast), dim=1)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  

        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size].view(-1,), num_classes=self.num_classes).to(torch.float32)
        one_hot_label = self.smooth / (self.num_classes - 1) * (1 - one_hot_label) + (1 - self.smooth) * one_hot_label

        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss