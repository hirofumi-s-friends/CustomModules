import pandas as pd
from enum import Enum
from azureml.studio.core.io.data_frame_directory import DataFrameDirectory, DataFrameSchema
from azureml.studio.internal.error import ErrorMapping
from azureml.designer.modules.recommendation.dnn.common.dataset import TransactionDataset, FeatureDataset
from azureml.designer.modules.recommendation.dnn.wide_and_deep.common.wide_n_deep_model import WideNDeepModel
from azureml.designer.modules.recommendation.dnn.wide_and_deep.score. \
    rating_prediction_scorer import RatingPredictionScorer
from azureml.designer.modules.recommendation.dnn.wide_and_deep.score. \
    recommend_rated_item_scorer import RecommendRatedItemScorer
from azureml.designer.modules.recommendation.dnn.wide_and_deep.score. \
    recommend_unrated_item_scorer import RecommendUnratedItemScorer
from azureml.designer.modules.recommendation.dnn.wide_and_deep.score.recommend_all_items_scorer import \
    RecommendAllItemScorer
from azureml.designer.modules.recommendation.dnn.common.entry_utils import params_loader
from azureml.designer.modules.recommendation.dnn.common.entry_param import Boolean
from azureml.designer.modules.recommendation.dnn.common.score_column_names import \
    build_rated_ranking_column_name_keys, build_ranking_column_name_keys


class RecommenderPredictionKind(Enum):
    RatingPrediction = "Rating Prediction"
    ItemRecommendation = "Item Recommendation"


class RecommendedItemSelection(Enum):
    FromAllItems = "From All Items"
    FromRatedItems = "From Rated Items (for model evaluation)"
    FromUnratedItems = "From Unrated Items (to suggest new items to users)"


class ScoreWideAndDeepRecommenderModule:
    def __init__(self):
        self.recommender_prediction_kind = None
        self.recommended_item_selection = None
        self.maximum_number_of_items_to_recommend_to_a_user = None
        self.minimum_size_of_the_recommendation_pool_for_a_single_user = None
        self.whether_to_return_the_predicted_ratings_of_the_items_along_with_the_labels = None

    @params_loader
    def on_init(self,
                trained_wide_and_deep_recommendation_model: WideNDeepModel,
                dataset_to_score: TransactionDataset,
                training_data: TransactionDataset,
                user_features: FeatureDataset,
                item_features: FeatureDataset,
                recommender_prediction_kind: RecommenderPredictionKind = None,
                recommended_item_selection: RecommendedItemSelection = None,
                maximum_number_of_items_to_recommend_to_a_user: int = None,
                minimum_size_of_the_recommendation_pool_for_a_single_user: int = None,
                whether_to_return_the_predicted_ratings_of_the_items_along_with_the_labels: Boolean = None):
        self.recommender_prediction_kind = recommender_prediction_kind
        self.recommended_item_selection = recommended_item_selection
        self.maximum_number_of_items_to_recommend_to_a_user = maximum_number_of_items_to_recommend_to_a_user
        self.minimum_size_of_the_recommendation_pool_for_a_single_user = \
            minimum_size_of_the_recommendation_pool_for_a_single_user
        self.whether_to_return_the_predicted_ratings_of_the_items_along_with_the_labels = \
            whether_to_return_the_predicted_ratings_of_the_items_along_with_the_labels

    @staticmethod
    def get_scorer(prediction_kind: RecommenderPredictionKind, recommended_item_selection: RecommendedItemSelection):
        if prediction_kind == RecommenderPredictionKind.RatingPrediction:
            return RatingPredictionScorer()
        elif prediction_kind == RecommenderPredictionKind.ItemRecommendation:
            if recommended_item_selection == RecommendedItemSelection.FromAllItems:
                return RecommendAllItemScorer()
            elif recommended_item_selection == RecommendedItemSelection.FromRatedItems:
                return RecommendRatedItemScorer()
            elif recommended_item_selection == RecommendedItemSelection.FromUnratedItems:
                return RecommendUnratedItemScorer()
            else:
                raise NotImplementedError(f"{recommended_item_selection} not supported now.")
        else:
            raise NotImplementedError(f"{prediction_kind} and {recommended_item_selection} not supported now.")

    def update_params(self,
                      recommender_prediction_kind: RecommenderPredictionKind = None,
                      recommended_item_selection: RecommendedItemSelection = None,
                      maximum_number_of_items_to_recommend_to_a_user: int = None,
                      minimum_size_of_the_recommendation_pool_for_a_single_user: int = None,
                      whether_to_return_the_predicted_ratings_of_the_items_along_with_the_labels: Boolean = None):
        for attr_name, attr_value in locals().items():
            if attr_name != "self" and attr_value is not None:
                setattr(self, attr_name, attr_value)

    @staticmethod
    def set_inputs_name(test_transactions: TransactionDataset, training_transactions: TransactionDataset = None,
                        user_features: FeatureDataset = None, item_features: FeatureDataset = None):
        _TRANSACTIONS_NAME = "Dataset to score"
        _USER_FEATURES_NAME = "User features"
        _ITEM_FEATURES_NAME = "Item features"
        _TRAINING_TRANSACTIONS_NAME = "Training data"
        if test_transactions is not None:
            test_transactions.name = _TRANSACTIONS_NAME
        else:
            ErrorMapping.verify_not_null_or_empty(x=test_transactions, name=_TRANSACTIONS_NAME)
        if training_transactions is not None:
            training_transactions.name = _TRAINING_TRANSACTIONS_NAME
        if user_features is not None:
            user_features.name = _USER_FEATURES_NAME
        if item_features is not None:
            item_features.name = _ITEM_FEATURES_NAME

    def build_scored_data_data_frame_directory(self, scored_data_df: pd.DataFrame):
        schema = DataFrameSchema.from_data_frame(scored_data_df)
        # if the scored_data_df contains predicted ratings columns, set score_column_names attr in the
        # DataFrameSchema to indicate score columns, to enable evaluate scored data with extra predicted ratings.
        if self.recommender_prediction_kind == RecommenderPredictionKind.ItemRecommendation and \
                self.whether_to_return_the_predicted_ratings_of_the_items_along_with_the_labels:
            if self.recommended_item_selection == RecommendedItemSelection.FromRatedItems:
                score_column_keys = build_rated_ranking_column_name_keys(
                    top_k=self.maximum_number_of_items_to_recommend_to_a_user)
            else:
                score_column_keys = build_ranking_column_name_keys(
                    top_k=self.maximum_number_of_items_to_recommend_to_a_user)
            scored_data_df_columns = scored_data_df.columns.to_list()
            score_column_keys_to_names = dict(
                zip(score_column_keys, scored_data_df_columns[:1] + scored_data_df_columns[1::2]))
            schema.score_column_names = score_column_keys_to_names

        schema = schema.to_dict()
        scored_data_dfd = DataFrameDirectory.create(data=scored_data_df, schema=schema)

        return scored_data_dfd

    @params_loader
    def run(self,
            trained_wide_and_deep_recommendation_model: WideNDeepModel,
            dataset_to_score: TransactionDataset,
            training_data: TransactionDataset,
            user_features: FeatureDataset,
            item_features: FeatureDataset,
            recommender_prediction_kind: RecommenderPredictionKind = None,
            recommended_item_selection: RecommendedItemSelection = None,
            maximum_number_of_items_to_recommend_to_a_user: int = None,
            minimum_size_of_the_recommendation_pool_for_a_single_user: int = None,
            whether_to_return_the_predicted_ratings_of_the_items_along_with_the_labels: Boolean = None,
            scored_data: str = None):
        self.update_params(recommender_prediction_kind, recommended_item_selection,
                           maximum_number_of_items_to_recommend_to_a_user,
                           minimum_size_of_the_recommendation_pool_for_a_single_user,
                           whether_to_return_the_predicted_ratings_of_the_items_along_with_the_labels)
        self.set_inputs_name(dataset_to_score, training_data, user_features=user_features,
                             item_features=item_features)
        scorer = self.get_scorer(self.recommender_prediction_kind, self.recommended_item_selection)
        scored_data_df = scorer.score(
            trained_wide_and_deep_recommendation_model,
            test_transactions=dataset_to_score,
            user_features=user_features,
            item_features=item_features,
            training_transactions=training_data,
            max_recommended_item_count=self.maximum_number_of_items_to_recommend_to_a_user,
            min_recommendation_pool_size=self.minimum_size_of_the_recommendation_pool_for_a_single_user,
            return_ratings=self.whether_to_return_the_predicted_ratings_of_the_items_along_with_the_labels
        )

        scored_data_dfd = self.build_scored_data_data_frame_directory(scored_data_df=scored_data_df)

        # scored_data is scored data output path, and the variable name is defined according to the module spec
        if scored_data is not None:
            scored_data_dfd.dump(save_to=scored_data)
        return scored_data_dfd,
