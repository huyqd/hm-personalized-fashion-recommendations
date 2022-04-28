import implicit
import numpy as np
import pandas as pd
from implicit.nearest_neighbours import bm25_weight

from config import Config
from dataset import convert_df_to_csr
from utils import Column

if __name__ == '__main__':
    ids = pd.read_csv("data/sample_submission.csv", usecols=[0]).squeeze().tolist()
    interaction_df = pd.read_csv("data/transactions_train.csv", dtype={Column.ARTICLE_ID: str})
    interaction_df[Column.N_INTERACTIONS] = 1
    interaction_df = interaction_df.groupby(
        [Column.CUSTOMER_ID, Column.ARTICLE_ID]
    )[Column.N_INTERACTIONS].sum().reset_index()

    csr, user_list, item_list = convert_df_to_csr(interaction_df)
    normalization_params = dict(K1=100, B=0.8)
    csr = bm25_weight(csr, **normalization_params).tocsr()

    # 2. Train model.
    embedding_dim = 40
    regularization = 0.1
    model = implicit.als.AlternatingLeastSquares(
        factors=embedding_dim,
        use_gpu=False,
        regularization=regularization,
        calculate_training_loss=True,
    )

    # The package takes in matrix in the form of items x users, hence transpose
    model.fit(csr)
    items, scores = model.recommend(np.arange(user_list.shape[0]), csr, N=Config.N_RECOMMENDATIONS)
    items = item_list.to_numpy()[items].astype(str)
    items = [" ".join(i.tolist()) for i in items]
    submission = pd.DataFrame({Column.CUSTOMER_ID: user_list, "prediction": items}).set_index(Column.CUSTOMER_ID)
    submission = submission.reindex(ids).ffill().reset_index()
    submission.to_csv("submission.csv", index=True)
