import numpy as np
import torch
import time
import json
from torch.utils.data import Dataset

def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    return us_pois, max_len


def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity


class Data(Dataset):
    def __init__(self, data, product_attributes, opt, train_len=None):
        inputs, max_len = handle_data(data[0], train_len)
        self.product_attributes = product_attributes
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.length = len(data[0])
        self.max_len = max_len
        self.sample_num = opt.sample_num
        self.attribute_kinds = opt.attribute_kinds
        self.dataset = opt.dataset

    def get_attribute_same_items(self, item, kind):
        as_items = self.product_attributes[f'{item}'][f'same_{kind}'].copy()
        # shuffle
        np.random.shuffle(as_items)
        if len(as_items) > self.sample_num:
            as_items = as_items[:self.sample_num]
        else:
            as_items += (self.sample_num - len(as_items)) * [0]

        return as_items

    def __getitem__(self, index):
        u_input, target = self.inputs[index], self.targets[index]

        max_n_node = self.max_len
        node = []
        neighbor_simi_mask = np.zeros([max_n_node, max_n_node])
        for i, it in enumerate(u_input):
            if it == 0:
                break
            if it not in node:
                node.append(it)
        node = np.array(node)
        items = node.tolist() + (max_n_node - len(node)) * [0]
        attribute_same_items = []
        attribute_same_items_SSL = []
        for _ in range(self.attribute_kinds):
            attribute_same_items.append([])
            attribute_same_items_SSL.append([])
        adj = np.zeros((max_n_node, max_n_node))
        last_item = u_input[0]
        for i in np.arange(len(u_input) - 1):
            # obtain adj
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3

        # obtain neighbor
        for i in np.arange(len(items)):
            if items[i] == 0 and i == 0:
                for k in range(self.attribute_kinds):
                    attribute_same_items[k].append(self.sample_num * [0])
                    attribute_same_items_SSL[k].append(self.sample_num * [0])
            elif items[i] == 0 and i != 0:
                for k in range(self.attribute_kinds):
                    attribute_same_items[k] += [self.sample_num * [0]] * (len(items) - i)
                    attribute_same_items_SSL[k] += [self.sample_num * [0]] * (len(items) - i)
                break
            else:
                neighbor_simi_mask[i][i] = 1
                if self.dataset == "Tmall":
                    cate_as_items = self.get_attribute_same_items(items[i], "cate")
                    attribute_same_items[0].append(cate_as_items)
                    cate_as_SSL = self.get_attribute_same_items(items[i], "cate")
                    attribute_same_items_SSL[0].append(cate_as_SSL)

                    brand_as_items = self.get_attribute_same_items(items[i], "brand")
                    attribute_same_items[1].append(brand_as_items)
                    brand_as_SSL = self.get_attribute_same_items(items[i], "brand")
                    attribute_same_items_SSL[1].append(brand_as_SSL)

                if self.dataset == "diginetica" or self.dataset == "30music":
                    cate_as_items = self.get_attribute_same_items(items[i], "cate")
                    attribute_same_items[0].append(cate_as_items)
                    cate_as_SSL = self.get_attribute_same_items(items[i], "cate")
                    attribute_same_items_SSL[0].append(cate_as_SSL)

        for k in range(self.attribute_kinds):
            attribute_same_items[k] = torch.tensor(np.array(attribute_same_items[k]))
            attribute_same_items_SSL[k] = torch.tensor(np.array(attribute_same_items_SSL[k]))
        last_item_mask = np.array(items) == last_item
        items = np.array(items)
        neighbor_simi_mask = np.array(neighbor_simi_mask)

        return [torch.tensor(adj), torch.tensor(items), torch.tensor(target),
                torch.tensor(last_item_mask), attribute_same_items,
                attribute_same_items_SSL, neighbor_simi_mask]

    def __len__(self):
        return self.length
