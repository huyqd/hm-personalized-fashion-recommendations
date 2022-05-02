import json

import implicit
import numpy as np
from implicit.nearest_neighbours import bm25_weight

from metrics import get_metrics
from utils import IOPath
from config import Config

if __name__ == '__main__':
    res = np.load(IOPath.INTERACTION_MATRIX / "data.npz", allow_pickle=True)
    train_csr, val_csr = res['train_csr'].item(), res['val_csr'].item()

    normalization_params = dict(K1=100, B=0.8)
    train_csr = bm25_weight(train_csr, **normalization_params).tocsr()

    # 2. Train model.
    embedding_dim = 40
    regularization = 0.1
    model = implicit.als.AlternatingLeastSquares(
        factors=embedding_dim,
        use_gpu=False,
        regularization=regularization,
        calculate_training_loss=True,
    )

    model.fit(train_csr)
    val_users = np.unique(val_csr.tocoo().row)
    val_items, val_scores = model.recommend(val_users, train_csr[val_users], N=Config.N_RECOMMENDATIONS)
    metrics = get_metrics(csr=val_csr, predictions=val_items)

    with open("metrics.json", "w") as fd:
        json.dump(metrics, fd, indent=4)
