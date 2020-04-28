import pandas as pd
from azureml.studio.core.logger import module_logger, TimeProfile
from azureml.studio.internal.error import ErrorMapping
from azureml.designer.modules.recommendation.dnn.common.dataset import Dataset, TransactionDataset, FeatureDataset
from azureml.designer.modules.recommendation.dnn.wide_and_deep.common.wide_n_deep_model import WideNDeepModel
from azureml.designer.modules.recommendation.dnn.common.constants import TRANSACTIONS_USER_COL
from azureml.designer.modules.recommendation.dnn.wide_and_deep.score. \
    base_recommender_scorer import BaseRecommenderScorer
from azureml.designer.modules.recommendation.dnn.common.score_column_names import build_ranking_column_names


class RecommendAllItemScorer(BaseRecommenderScorer):
    def _validate_parameters(self, learner: WideNDeepModel, test_data: Dataset, user_features: FeatureDataset = None,
                             item_features: FeatureDataset = None, **kwargs):
        super()._validate_parameters(learner, test_data, user_features=user_features, item_features=item_features)
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=test_data.column_size,
                                                                       required_column_count=1,
                                                                       arg_name=test_data.name)

    def _preprocess(self, transactions: TransactionDataset, user_features: FeatureDataset = None,
                    item_features: FeatureDataset = None, training_transactions: TransactionDataset = None):
        # remove extra items/ratings column
        if transactions.users is not None:
            transactions.df = pd.DataFrame(transactions.users)
            transactions.build_column_attributes()
        return super()._preprocess(transactions, user_features, item_features, training_transactions)

    def score(self, learner: WideNDeepModel, test_transactions: TransactionDataset,
              user_features: FeatureDataset = None,
              item_features: FeatureDataset = None, **kwargs):
        module_logger.info("Recommendation task: Recommend items from all item.")
        super().score(learner, test_transactions, user_features, item_features, **kwargs)
        max_recommended_item_count = kwargs["max_recommended_item_count"]
        return_ratings = kwargs["return_ratings"]
        all_items = learner.item_feature_builder.id_vocab
        test_transactions_df = test_transactions.df
        users = test_transactions_df.iloc[:, TRANSACTIONS_USER_COL].unique()
        module_logger.info(f"Get {len(users)} unique users, and {len(all_items)} unique items.")

        with TimeProfile("Building complete user item transactions dataset"):
            transactions_df = self.build_user_item_cartesian_pairs(users=users, items=all_items)
        transactions = TransactionDataset(transactions_df)
        recommendations = self._recommend(learner, transactions=transactions, K=max_recommended_item_count,
                                          user_features=user_features, item_features=item_features)
        return self._format_recommendations(recommendations, return_ratings, K=max_recommended_item_count,
                                            score_column_names_build_method=build_ranking_column_names)
