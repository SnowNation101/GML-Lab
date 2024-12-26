import random
import numpy as np
import jittor

import os
import pandas as pd

from .construct_subgraph import construct_mini_batch_giant_graph, get_parallel_sampler, get_mini_batch

import torch_sparse
# from jittor.sparse import SparseVar,spmm
from jittor_geometric.nn.conv.gcn_conv import gcn_norm

from tqdm import tqdm
import pickle
##############################################################################
##############################################################################
##############################################################################
# utility function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # jittor.manual_seed(seed)
    # jittor.cuda.manual_seed_all(seed)

def sparse_tensor_to_sparse_var(sparse_tensor):
    # 提取行、列和值
    row, col, value = sparse_tensor.coo()
    
    # 将数据转换为 jt.Var
    row = jittor.array(row.numpy())
    col = jittor.array(col.numpy())
    value = jittor.array(value.numpy())
    
    # 将 indices 转换为 jt.Var
    indices = jittor.stack([row, col])
    
    # 获取形状
    shape = jittor.NanoVector([sparse_tensor.size(0), sparse_tensor.size(1)])
    
    # 创建 SparseVar 对象
    sparse_var = jittor.sparse.SparseVar(indices, value, shape)
    
    return sparse_var
def dense_to_sparse(dense_matrix):
    # 获取非零元素的索引
    indices = jittor.nonzero(dense_matrix)
    values = dense_matrix[indices[:, 0], indices[:, 1]]
    shape = jittor.NanoVector([int(dense_matrix.shape[0]), int(dense_matrix.shape[1])])
    return jittor.sparse.SparseVar(indices.transpose(), values, shape) 
  

def row_norm(adj_t):
    if isinstance(adj_t, torch_sparse.SparseTensor):
    # if isinstance(adj_t, SparseVar):
        # adj_t = jittor_sparse.fill_diag(adj, 1)
        print("adj_t",adj_t)
        # pickle.dump(adj_t, open("adj_t.pkl", "wb"))
        adj_t = sparse_tensor_to_sparse_var(adj_t)
        # deg = torch_sparse.sum(adj_t, dim=1)
        deg = adj_t.to_dense().sum(dim=1)
        #jittor: deg = adj_t.to_dense().sum(axis=1)
        print("deg",deg)
        # pickle.dump(deg, open("deg.pkl", "wb"))
        # deg = sum(adj_t, dim=1)
        # deg = adj_t.sum(axis=1)
        deg_inv = 1. / deg
        print("deg_inv",deg_inv)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        print("deg_inv",deg_inv)
        # pickle.dump(deg_inv, open("deg_inv.pkl", "wb"))
        # adj_t = torch_sparse.mul(adj_t, deg_inv.view(-1, 1))
        adj_t = dense_to_sparse(adj_t.to_dense().mul(deg_inv.view(-1, 1)))
        print("adj_t",adj_t)
        # adj_t = spmm(adj_t, deg_inv.view(-1, 1))、
        # pickle.dump(adj_t, open("adj_t1.pkl", "wb"))
        return adj_t

def sym_norm(adj):
    if isinstance(adj, torch_sparse.SparseTensor):
    # if isinstance(adj, SparseVar):
        adj_t = gcn_norm(adj, add_self_loops=False) 
        return adj_t

##############################################################################
##############################################################################
##############################################################################
# load data

def load_feat(d):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = jittor.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == jittor.bool:
            node_feats = node_feats.type(jittor.float32)

    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = jittor.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == jittor.bool:
            edge_feats = edge_feats.type(jittor.float32)

    return node_feats, edge_feats    

def load_graph(d):
    df = pd.read_csv('GraphMixer/DATA/{}/edges.csv'.format(d))
    g = np.load('GraphMixer/DATA/{}/ext_full.npz'.format(d))

    # df = pd.read_csv('data/{}/raw/{}.csv'.format(d, d))
    # g = np.load('data/{}/processed/ext_full.npz'.format(d))
    return g, df

def node_cls_info(args):
    # load node label information
    ldf = pd.read_csv('DATA/{}/labels.csv'.format(args.data))
    node_role = jittor.tensor(ldf['ext_roll'].values, dtype=jittor.int32)
    node_labels = jittor.tensor(ldf['label'].values, dtype=jittor.int32)
    
    return ldf, node_role, node_labels

##############################################################################
##############################################################################
##############################################################################

@jittor.no_grad()
def get_node_embeds(model, edge_feats, g, df, args):
    # for each node, sample its neighbors with the most recent neighbors (sorted) 
    sampler, neg_link_sampler = get_parallel_sampler(g, args.num_neighbors)

    loader = df.groupby(df.index // args.batch_size)
    pbar = tqdm(total=len(loader))
    pbar.set_description('Compute node embeddings ...')

    ###################################################
    all_embds = []

    sampler.reset()
    for _, rows in loader:
        # root_nodes = [edge_src_node, edge_dst_node, random_neg_nodes] of size 3 * batch_size
        root_nodes = np.array(rows.node.values, dtype=np.int32)
        ts = np.array(rows.time.values, dtype=np.float32)

        # get subgraph data
        inputs = sampler, root_nodes, ts, args.sampled_num_hops, 0
        subgraph_data = get_mini_batch(*inputs)
        subgraph_data = construct_mini_batch_giant_graph(subgraph_data, args.max_edges)

        # raw edge feats 
        subgraph_edge_feats = edge_feats[subgraph_data['eid']]
        subgraph_edts = jittor.array(subgraph_data['edts']).float32()

        # get mini-batch inds
        all_inds, has_temporal_neighbors = [], []

        # ignore an edge pair if (src_node, dst_node) does not have temporal neighbors
        all_edge_indptr = subgraph_data['all_edge_indptr']
        for i in range(len(all_edge_indptr)-1):
            num_edges = all_edge_indptr[i+1] - all_edge_indptr[i]
            all_inds.extend([(args.max_edges * i + j) for j in range(num_edges)])
            has_temporal_neighbors.append(num_edges>0)

        ###################################################
        inputs = [
            subgraph_edge_feats.to(args.device), 
            subgraph_edts.to(args.device), 
            len(has_temporal_neighbors), 
            jittor.tensor(all_inds).long()
        ]

        cur_embs = model(*inputs).clone().detach().cpu()
        all_embds.append(cur_embs)

        pbar.update(1)
    pbar.close()
    
    all_embds = jittor.cat(all_embds, dim=0)
    return all_embds

###################################################
# compute hits@K score
def hits_at_K(y_pred_pos, y_pred_neg, K=50):
    y_pred_pos, y_pred_neg = y_pred_pos.flatten().detach(), y_pred_neg.flatten().detach()
    kth_score_in_negative_edges = jittor.topk(y_pred_neg, K)[0][-1]
    hitsK = float(jittor.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
    return hitsK