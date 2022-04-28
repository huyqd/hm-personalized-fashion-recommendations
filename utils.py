from pathlib import Path
import numpy as np

CURRENT_PATH = Path(__file__).cwd()


class Column:
    CUSTOMER_ID = "customer_id"
    ARTICLE_ID = "article_id"
    N_INTERACTIONS = "n_interactions"
    DATE = "t_dat"


class IOPath:
    DATA = CURRENT_PATH / 'data'
    INTERACTION_MATRIX = DATA / 'interaction_matrix'

    ARTICLES = DATA / 'articles.csv'
    ARTICLES_MAPPING = DATA / 'articles_mapping.json'
    CUSTOMERS = DATA / 'customers.csv'
    CUSTOMERS_MAPPING = DATA / 'customers_mapping.json'
    TRANSACTIONS_TRAIN = DATA / 'transactions_train.csv'
    SAMPLE_SUBMISSION = DATA / 'sample_submission.csv'
    SUBMISSION = DATA / 'submission.csv'


def find_matches(csr, predictions):
    matches = []
    val_users = np.unique(csr.tocoo().row)
    for idx, user in enumerate(val_users):
        matches.append(csr[user].toarray().squeeze()[predictions[idx]].reshape(1, -1))

    matches = np.concatenate(matches, axis=0)

    return matches
