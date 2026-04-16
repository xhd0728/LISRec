import torch
import numpy as np


def recall(pos_index, pos_len):
    return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


def ndcg(pos_index, pos_len):
    rank_length = np.full_like(pos_len, pos_index.shape[1])
    ideal_length = np.where(pos_len > rank_length, rank_length, pos_len)
    positions = np.arange(1, pos_index.shape[1] + 1, dtype=np.float32)

    ideal_ranks = np.broadcast_to(positions, pos_index.shape)
    idcg = np.cumsum(1.0 / np.log2(ideal_ranks + 1), axis=1)
    for row_index, cutoff in enumerate(ideal_length):
        if cutoff > 0:
            idcg[row_index, cutoff:] = idcg[row_index, cutoff - 1]

    ranks = np.broadcast_to(positions, pos_index.shape)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    return dcg / idcg


def get_metrics_dict(rank_indices, n_seq, n_item, Ks, target_item_list):
    rank_indices = torch.as_tensor(rank_indices, dtype=torch.long)
    pos_matrix = torch.zeros((n_seq, n_item), dtype=torch.int64)
    for row_index, target_item in enumerate(target_item_list):
        pos_matrix[row_index, target_item - 1] = 1

    pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
    pos_idx = torch.gather(pos_matrix, dim=1, index=rank_indices)
    pos_idx = pos_idx.to(torch.bool).cpu().numpy()
    pos_len_list = pos_len_list.squeeze(-1).cpu().numpy()

    recall_result = recall(pos_idx, pos_len_list)
    avg_recall_result = recall_result.mean(axis=0)
    ndcg_result = ndcg(pos_idx, pos_len_list)
    avg_ndcg_result = ndcg_result.mean(axis=0)

    metrics_dict = {}
    for topk in Ks:
        metrics_dict[topk] = {
            "recall": round(avg_recall_result[topk - 1], 4),
            "ndcg": round(avg_ndcg_result[topk - 1], 4),
        }

    return metrics_dict
