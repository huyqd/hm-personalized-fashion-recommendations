import numpy as np

from config import Config
from utils import find_matches


def get_hr(matches):
    return (matches.sum(axis=1) / matches.shape[1] * 100).mean()


def get_mapak(matches, denominator):
    # compute average precision at each K
    p_at_k_numerator = (matches.cumsum(axis=1) * matches) / np.arange(1, Config.N_RECOMMENDATIONS + 1)
    # compute average precision over all Ks
    apak = p_at_k_numerator.sum(axis=1) / denominator * 100
    mapak = apak.mean()

    return mapak


# def get_ndcg(matches):
#     pass


def get_metrics(csr=None, predictions=None):
    matches = find_matches(csr=csr, predictions=predictions)
    apak_denom = np.minimum(csr[np.unique(csr.tocoo().row)].sum(axis=1), Config.N_RECOMMENDATIONS).squeeze().A

    return {
        "hr": get_hr(matches),
        "mapak": get_mapak(matches, apak_denom),
        # "ndcg": get_ndcg(matches),
    }
