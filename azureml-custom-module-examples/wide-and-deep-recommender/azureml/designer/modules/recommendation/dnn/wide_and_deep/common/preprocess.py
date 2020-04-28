import numpy as np
from azureml.designer.modules.recommendation.dnn.common.dataset import TransactionDataset, FeatureDataset
from azureml.designer.modules.recommendation.dnn.common.utils import convert_to_str
from azureml.studio.core.logger import time_profile


@time_profile
def preprocess_transactions(transactions: TransactionDataset):
    """Preprocess transaction dataset.

    The preprocess including:
    1. Drop instances with missing values
    2. Convert user/item ids to string type
    """
    if transactions.ratings is not None:
        transactions.ratings = transactions.ratings.replace(to_replace=[np.inf, -np.inf], value=np.nan)
    if transactions.users is not None:
        transactions.users = convert_to_str(transactions.users)
    if transactions.items is not None:
        transactions.items = convert_to_str(transactions.items)

    transactions.df = transactions.df.dropna().reset_index(drop=True)
    transactions.build_column_attributes()

    return transactions


@time_profile
def preprocess_features(features: FeatureDataset):
    """Preprocess feature dataset.

    The preprocess including:
    1. Drop instances with missing values
    2. Convert user/item ids to string type
    """
    features.df = features.df.dropna(subset=[features.ids.name]).reset_index(drop=True)
    features.ids = convert_to_str(features.ids)
    features.build_column_attributes()

    return features
