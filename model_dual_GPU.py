import datetime
import numpy as np
from tqdm import tqdm
from aggregator import *
from torch.nn import Module
import torch.nn.functional as F

# device conf
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

class CombineGraph(Module):
    def __init__(self, opt, num_node):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.hop = opt.n_iter
        self.mu = opt.mu

        # Aggregator
        self.attribute_agg = AttributeAggregator(self.dim, self.opt.alpha, opt, self.opt.dropout_attribute)
        self.local_agg = nn.ModuleList()
        self.mirror_agg = nn.ModuleList()
        for i in range(self.hop):
            agg = LocalAggregator(self.dim, self.opt.alpha)
            self.local_agg.append(agg)
            agg = MirrorAggregator(self.dim)
            self.mirror_agg.append(agg)

        # high way net
        self.highway = nn.Linear(self.dim * 2, self.dim, bias=False)

        # embeddings
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu3 = nn.Linear(self.dim, self.dim)
        self.gate = nn.Linear(self.dim * 2, self.dim, bias=False)

        # loss function
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_score(self, hidden, pos_emb, h_mirror, h_local, mask, item_weight):
        hm = h_mirror
        hl = h_local.unsqueeze(1).repeat(1, hidden.size(1), 1)
        hp = hidden + pos_emb
        nh = torch.sigmoid(self.glu1(hp) + self.glu2(hm) + self.glu3(hl))
        beta = torch.matmul(nh, self.w)
        beta = beta * mask
        zg = torch.sum(beta * hp, 1)
        gf = torch.sigmoid(self.gate(torch.cat([zg, h_local], dim=-1))) * self.mu
        zh = gf * h_local + (1 - gf) * zg
        zh = F.dropout(zh, self.opt.dropout_score, self.training)
        scores = torch.matmul(zh, item_weight.transpose(1, 0))

        return scores

    def similarity_loss(self, hf, hf_SSL, simi_mask):
        h1 = hf
        h2 = hf_SSL
        h1 = h1.unsqueeze(2).repeat(1, 1, h1.size(1), 1)
        h2 = h2.unsqueeze(1).repeat(1, h2.size(1), 1, 1)
        hf_similarity = torch.sum(torch.mul(h1, h2), dim=3) / self.opt.temp
        loss = -torch.log(torch.softmax(hf_similarity, dim=2) + 1e-8)
        simi_mask = simi_mask == 1
        loss = torch.sum(loss * simi_mask, dim=2)
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)

        return loss

    def compute_score_and_ssl_loss(self, h, h_local, h_mirror, mask, hf_SSL1, hf_SSL2, simi_mask):
        mask = mask.float().unsqueeze(-1)
        batch_size = h.shape[0]
        len = h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        b = self.embedding.weight[1:]

        simi_loss = self.similarity_loss(hf_SSL1, hf_SSL2, simi_mask)
        scores = self.compute_score(h, pos_emb, h_mirror, h_local, mask, b)

        return simi_loss, scores

    def forward(self, inputs, adj, last_item_mask, as_items, as_items_SSL, simi_mask):
        # preprocess
        mask_item = inputs != 0
        attribute_num = len(as_items)
        h = self.embedding(inputs)
        h_as = []
        h_as_SSL = []
        as_mask = []
        as_mask_SSL = []
        for k in range(attribute_num):
            nei = as_items[k]
            nei_SSL = as_items_SSL[k]
            nei_emb = self.embedding(nei)
            nei_emb_SSL = self.embedding(nei_SSL)
            h_as.append(nei_emb)
            h_as_SSL.append(nei_emb_SSL)
            as_mask.append(as_items[k] != 0)
            as_mask_SSL.append(as_items_SSL[k] != 0)

        # attribute
        hf_1, hf_2, hf = self.attribute_agg(h, h_as, as_mask, h_as_SSL, as_mask_SSL)

        # GNN
        x = h
        mirror_nodes = hf
        for i in range(self.hop):
            # aggregate neighbor info
            x = self.local_agg[i](x, adj, mask_item)
            x, mirror_nodes = self.mirror_agg[i](mirror_nodes, x, mask_item)

        # highway
        g = torch.sigmoid(self.highway(torch.cat([h, x], dim=2)))
        x_dot = g * h + (1 - g) * x

        # hidden
        hidden = x_dot

        # local representation
        h_local = torch.masked_select(x_dot, last_item_mask.unsqueeze(2).repeat(1, 1, x_dot.size(2))).reshape(mask_item.size(0), -1)

        # mirror
        h_mirror = mirror_nodes

        # calculate score
        simi_loss, scores = self.compute_score_and_ssl_loss(hidden, h_local, h_mirror, mask_item, hf_1, hf_2, simi_mask)

        return simi_loss, scores


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.to(device)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, opt):
    adj, items, targets, last_item_mask, as_items, as_items_SSL, simi_mask = data
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    last_item_mask = trans_to_cuda(last_item_mask)
    for k in range(opt.attribute_kinds):
        as_items[k] = trans_to_cuda(as_items[k]).long()
        as_items_SSL[k] = trans_to_cuda(as_items_SSL[k]).long()
    targets_cal = trans_to_cuda(targets).long()
    simi_mask = trans_to_cuda(simi_mask).long()

    simi_loss, scores = model(items, adj, last_item_mask, as_items, as_items_SSL, simi_mask)
    simi_loss = torch.mean(simi_loss)
    loss = model.module.loss_function(scores, targets_cal - 1)
    loss = loss + opt.phi * simi_loss

    return targets, scores, loss

def adjust_learning_rate(optimizer, decay_rate, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * decay_rate
    lr * decay_rate

def train_test(model, opt, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=opt.batch_size,
                                               shuffle=True, pin_memory=False)
    for i, data in enumerate(tqdm(train_loader)):
        targets, scores, loss = forward(model, data, opt)
        loss.backward()
        model.module.optimizer.step()
        model.module.optimizer.zero_grad()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    if opt.decay_count < opt.decay_num:
        model.module.scheduler.step()
        opt.decay_count += 1

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=int(opt.batch_size / 8),
                                              shuffle=False, pin_memory=False)
    result_20 = []
    hit_20, mrr_20 = [], []
    result_10 = []
    hit_10, mrr_10 = [], []
    for data in test_loader:
        targets, scores, loss = forward(model, data, opt)
        targets = targets.numpy()
        sub_scores_20 = scores.topk(20)[1]
        sub_scores_20 = trans_to_cpu(sub_scores_20).detach().numpy()
        for score, target in zip(sub_scores_20, targets):
            hit_20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_20.append(0)
            else:
                mrr_20.append(1 / (np.where(score == target - 1)[0][0] + 1))

        sub_scores_10 = scores.topk(10)[1]
        sub_scores_10 = trans_to_cpu(sub_scores_10).detach().numpy()
        for score, target in zip(sub_scores_10, targets):
            hit_10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_10.append(0)
            else:
                mrr_10.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result_20.append(np.mean(hit_20) * 100)
    result_20.append(np.mean(mrr_20) * 100)

    result_10.append(np.mean(hit_10) * 100)
    result_10.append(np.mean(mrr_10) * 100)

    return result_10, result_20
