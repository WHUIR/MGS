import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy
import math

class AttributeAggregator(nn.Module):
    def __init__(self, dim, alpha, opt, dropout):
        super(AttributeAggregator, self).__init__()
        self.dropout = dropout
        self.dim = dim
        self.opt = opt
        self.Ws = nn.ModuleList()
        for k in range(opt.attribute_kinds):
            w = nn.Linear(self.dim * 2, 1, bias=False)
            self.Ws.append(w)
        self.combine_NN = nn.Linear(self.dim * opt.attribute_kinds, self.dim, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def get_weight(self, hidden, as_items):
        # calculate attribute-same items' attention weights
        neighbor_num = as_items[0].size(2)
        hidden_f = hidden.unsqueeze(2).repeat(1, 1, neighbor_num, 1)
        E = []
        for k in range(self.opt.attribute_kinds):
            neighbor = as_items[k]
            e = self.Ws[k](torch.cat([hidden_f, neighbor], dim=3)).squeeze(-1)
            e = self.leakyrelu(e)
            E.append(e)

        return E

    def generate_mirror(self, alpha, as_items, attribute_mask):
        as_representations = []
        for i in range(self.opt.attribute_kinds):
            # preprocess
            _attribute_mask = torch.sum(attribute_mask[i], dim=2)
            _attribute_mask = _attribute_mask == 0
            _attribute_mask = _attribute_mask.unsqueeze(2).repeat(1, 1, attribute_mask[i].size(2))
            attribute_mask[i][_attribute_mask] = True

            a = alpha[i].clone()
            a[~attribute_mask[i]] = float('-inf')
            a = torch.softmax(a, dim=2).unsqueeze(3)

            # calculate representation
            hf_dot = torch.sum(a * as_items[i], dim=2)
            as_representations.append(hf_dot)

        hf = torch.cat([h for h in as_representations], dim=-1)
        if self.opt.attribute_kinds > 1:
            hf = self.combine_NN(hf)
            hf = F.dropout(hf, self.dropout, self.training)

        return hf

    def forward(self, hidden, as_items_1, as_mask_1, as_items_2, as_mask_2):
        E_1 = self.get_weight(hidden, as_items_1)
        E_2 = self.get_weight(hidden, as_items_2)

        # generate 2 separate mirror graphs for SSL
        hf_1 = self.generate_mirror(E_1, as_items_1, as_mask_1)
        hf_2 = self.generate_mirror(E_2, as_items_2, as_mask_2)

        # merge two sets of as-item and generate the main mirror graph
        E = []
        as_items = []
        as_mask = []
        for i in range(self.opt.attribute_kinds):
            E.append(torch.cat([E_1[i], E_2[i]], dim=-1))
            as_items.append(torch.cat([as_items_1[i], as_items_2[i]], dim=2))
            as_mask.append(torch.cat([as_mask_1[i], as_mask_2[i]], dim=2))

        hf = self.generate_mirror(E, as_items, as_mask)

        return hf_1, hf_2, hf

class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.linear0 = nn.Linear(self.dim, 1, bias=False)
        self.linear1 = nn.Linear(self.dim, 1, bias=False)
        self.linear2 = nn.Linear(self.dim, 1, bias=False)
        self.linear3 = nn.Linear(self.dim, 1, bias=False)
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = self.linear0(a_input)
        e_1 = self.linear1(a_input)
        e_2 = self.linear2(a_input)
        e_3 = self.linear3(a_input)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = alpha.clone()
        mask_item = mask_item.unsqueeze(1).repeat(1, alpha.size(2), 1)
        alpha[~mask_item] = float('-inf')
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output

class MirrorAggregator(nn.Module):
    def __init__(self, dim):
        super(MirrorAggregator, self).__init__()
        self.dim = dim
        self.Wq1 = nn.Linear(self.dim, self.dim, bias=False)
        self.Wk1 = nn.Linear(self.dim, self.dim, bias=False)

        self.Wq2 = nn.Linear(self.dim, self.dim, bias=False)
        self.Wk2 = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, mirror_nodes, satellite_nodes, satellite_node_mask):
        alpha_left = self.Wq1(satellite_nodes).unsqueeze(2)
        alpha_right = self.Wk1(mirror_nodes).unsqueeze(3)
        alpha = torch.matmul(alpha_left, alpha_right) / math.sqrt(self.dim)
        alpha = alpha.squeeze().unsqueeze(2)
        satellite_nodes = satellite_nodes + alpha * (mirror_nodes - satellite_nodes)

        beta_left = self.Wq2(mirror_nodes)
        beta_right = self.Wk2(satellite_nodes).transpose(1, 2)
        beta = torch.matmul(beta_left, beta_right)
        beta = beta / math.sqrt(self.dim)
        satellite_node_mask = satellite_node_mask.unsqueeze(1).repeat(1, satellite_nodes.size(1), 1)
        beta = beta.clone()
        beta[~satellite_node_mask] = float('-inf')
        beta = F.softmax(beta, 2)
        mirror_nodes = torch.matmul(beta, mirror_nodes)

        return satellite_nodes, mirror_nodes
