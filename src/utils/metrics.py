from typing import List, Dict

import numpy as np


def get_metrics(targets: List[int], candidates: List[int], topk: int) -> Dict[str, float]:
    def I_k(cand) -> int:
        return int(cand in targets)
    hitrate = int(sum([I_k(cand) for cand in candidates]) > 0)
    recall = sum([I_k(cand) for cand in candidates]) / min(topk, len(targets))
    dcg = sum([I_k(candidates[k-1])/np.log2(k+1) for k in range(1, topk+1)])
    idcg = sum([1/np.log2(k+1) for k in range(1, min(topk+1, len(targets)+1))])
    ndcg = dcg/idcg
    return {
        "hitrate": hitrate,
        "recall": recall,
        "ndcg": ndcg,
    }


def evaluate(
    targets: Dict[int, List[int]],
    candidates: Dict[int, List[int]],
    catalog_size: int,
    topk: int = 100,
) -> Dict[str, float]:
    users_metrics = []
    for uid in targets:
        user_target = targets[uid]
        user_candidates = candidates[uid]
        users_metrics.append(get_metrics(user_target, user_candidates, topk))


    hitrate = np.mean([metrics['hitrate'] for metrics in users_metrics])
    recall = np.mean([metrics['recall'] for metrics in users_metrics])
    ndcg = np.mean([metrics['ndcg'] for metrics in users_metrics])

    candidates_set = []
    for cands in candidates.values():
        candidates_set.extend(cands)
    coverage = len(set(candidates_set)) / catalog_size
    return {
        "hitrate": hitrate,
        "recall": recall,
        "ndcg": ndcg,
        "coverage": coverage,
    }