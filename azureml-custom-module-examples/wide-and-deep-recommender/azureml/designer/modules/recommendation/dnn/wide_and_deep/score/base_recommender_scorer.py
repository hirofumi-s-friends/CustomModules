import numpy as np
import pandas as pd
from abc import abstractmethod
from azureml.studio.core.logger import TimeProfile
from azureml.studio.internal.error import ErrorMapping, DuplicateFeatureDefinitionError, InvalidColumnTypeError
from azureml.studio.core.data_frame_schema import ColumnTypeName
from azureml.designer.modules.recommendation.dnn.common.dataset import Dataset, TransactionDataset, FeatureDataset
from azureml.designer.modules.recommendation.dnn.wide_and_deep.common.wide_n_deep_model import WideNDeepModel
from azureml.designer.modules.recommendation.dnn.common.constants import TRANSACTIONS_RATING_COL
from azureml.designer.modules.recommendation.dnn.wide_and_deep.common.preprocess import preprocess_features, \
    preprocess_transactions
from azureml.designer.modules.recommendation.dnn.common.score_column_names import USER_COLUMN, ITEM_COLUMN, \
    SCORED_RATING


class BaseRecommenderScorer:
    def _validate_parameters(self, learner: WideNDeepModel, test_data: Dataset, user_features: FeatureDataset = None,
                             item_features: FeatureDataset = None, **kwargs):
        ErrorMapping.verify_not_null_or_empty(x=learner, name=WideNDeepModel.MODEL_NAME)
        ErrorMapping.verify_number_of_rows_greater_than_or_equal_to(curr_row_count=test_data.row_size,
                                                                    required_row_count=1,
                                                                    arg_name=test_data.name)
        ErrorMapping.verify_number_of_columns_less_than_or_equal_to(curr_column_count=test_data.column_size,
                                                                    required_column_count=3,
                                                                    arg_name=test_data.name)
        if user_features is not None:
            self._validate_feature_dataset(user_features)
        if item_features is not None:
            self._validate_feature_dataset(item_features)

    @staticmethod
    def _validate_features_type(dataset: FeatureDataset):
        for col in dataset.columns:
            if dataset.get_column_type(col) == ColumnTypeName.NAN:
                ErrorMapping.throw(InvalidColumnTypeError(col_type=dataset.get_column_type(col),
                                                          col_name=col,
                                                          arg_name=dataset.name))

    def _validate_feature_dataset(self, dataset: FeatureDataset):
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=dataset.column_size,
                                                                       required_column_count=2,
                                                                       arg_name=dataset.name)
        ErrorMapping.verify_number_of_rows_greater_than_or_equal_to(curr_row_count=dataset.row_size,
                                                                    required_row_count=1,
                                                                    arg_name=dataset.name)
        self._validate_features_type(dataset)

    def _preprocess(self, transactions: TransactionDataset, user_features: FeatureDataset = None,
                    item_features: FeatureDataset = None, training_transactions: TransactionDataset = None):
        transactions = preprocess_transactions(transactions)
        user_features = preprocess_features(user_features) if user_features is not None else None
        item_features = preprocess_features(item_features) if item_features is not None else None
        training_transactions = (
            preprocess_transactions(training_transactions) if training_transactions is not None else None
        )

        BaseRecommenderScorer._validate_preprocessed_dataset(user_features=user_features, item_features=item_features)

        return transactions, user_features, item_features, training_transactions

    @staticmethod
    def _validate_preprocessed_dataset(user_features: FeatureDataset, item_features: FeatureDataset):
        if user_features is not None and any(user_features.df.duplicated(subset=user_features.ids.name)):
            ErrorMapping.throw(DuplicateFeatureDefinitionError())
        if item_features is not None and any(item_features.df.duplicated(subset=item_features.ids.name)):
            ErrorMapping.throw(DuplicateFeatureDefinitionError())

    @abstractmethod
    def score(self, learner: WideNDeepModel, test_transactions: TransactionDataset,
              user_features: FeatureDataset = None,
              item_features: FeatureDataset = None, **kwargs):
        self._validate_parameters(learner=learner, test_data=test_transactions, user_features=user_features,
                                  item_features=item_features, **kwargs)
        training_transactions = kwargs['training_transactions']
        self._preprocess(transactions=test_transactions, user_features=user_features, item_features=item_features,
                         training_transactions=training_transactions)

    def _predict(self, learner: WideNDeepModel, transactions: TransactionDataset,
                 user_features: FeatureDataset = None,
                 item_features: FeatureDataset = None):
        learner.update_feature_builders(user_features=user_features, item_features=item_features)
        transactions = TransactionDataset(df=transactions.df.iloc[:, :TRANSACTIONS_RATING_COL].reset_index(drop=True))
        predictions = learner.predict(transactions)
        result_df = transactions.df
        result_df = result_df.rename(columns=dict(zip(result_df.columns, [USER_COLUMN, ITEM_COLUMN])))
        result_df[SCORED_RATING] = predictions
        return result_df

    def _recommend(self, learner: WideNDeepModel, transactions: TransactionDataset, K: int,
                   user_features: FeatureDataset = None, item_features: FeatureDataset = None):
        if transactions.row_size == 0:
            return pd.DataFrame(columns=[USER_COLUMN, ITEM_COLUMN, SCORED_RATING])
        predict_df = self._predict(learner, transactions, user_features=user_features, item_features=item_features)
        with TimeProfile(f"Get top {K} items for each user"):
            predict_df = (
                predict_df.groupby(by=[USER_COLUMN]).apply(
                    lambda x: x.nlargest(columns=SCORED_RATING, n=K)).reset_index(drop=True)
            )
            topK_items = predict_df.groupby(USER_COLUMN)[ITEM_COLUMN].apply(
                lambda x: (list(x) + [None] * K)[:K])
            topK_ratings = predict_df.groupby(USER_COLUMN)[SCORED_RATING].apply(
                lambda x: (list(x) + [0] * K)[:K])
        return pd.DataFrame(
            {USER_COLUMN: topK_items.index, ITEM_COLUMN: topK_items.values, SCORED_RATING: topK_ratings.values})

    def _format_recommendations(self, recommendations: pd.DataFrame, return_ratings: bool, K: int,
                                score_column_names_build_method):
        items = np.array(list(recommendations[ITEM_COLUMN])).reshape([-1, K])
        ratings = np.array(list(recommendations[SCORED_RATING])).reshape([-1, K])

        score_column_names = score_column_names_build_method(top_k=K)
        users_df = pd.DataFrame({score_column_names[0]: recommendations[USER_COLUMN]})
        recommended_items_df = pd.DataFrame(items, columns=score_column_names[1:])

        if return_ratings:
            score_column_names = self._insert_pred_rating_column_names(score_column_names)
            recommended_item_ratings_df = pd.DataFrame(ratings, columns=score_column_names[2::2])
            res_df = pd.concat([users_df, recommended_items_df, recommended_item_ratings_df], axis=1)
        else:
            res_df = pd.concat([users_df, recommended_items_df], axis=1)

        res_df = res_df[score_column_names]
        return res_df

    @staticmethod
    def _insert_pred_rating_column_names(score_column_names):
        PredRatingColumn = "Predicted Rating"
        top_k = len(score_column_names) - 1
        new_score_column_names = score_column_names[:1]
        for i in range(1, top_k + 1):
            new_score_column_names = new_score_column_names + [score_column_names[i], f"{PredRatingColumn} {i}"]
        return new_score_column_names

    @staticmethod
    def build_user_item_cartesian_pairs(users, items):
        users = np.array(users)
        items = np.array(items)
        r_users = np.repeat(users, len(items))
        r_items = items[np.newaxis, :]
        r_items = np.repeat(r_items, len(users), axis=0).flatten()
        return pd.DataFrame({USER_COLUMN: r_users, ITEM_COLUMN: r_items})
