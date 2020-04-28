from azureml.studio.internal.error import ErrorMapping, MoreThanOneRatingError, DuplicateFeatureDefinitionError, \
    InvalidDatasetError, InvalidColumnTypeError
from azureml.studio.core.data_frame_schema import ColumnTypeName
from azureml.designer.modules.recommendation.dnn.common.constants import TRANSACTIONS_RATING_COL, \
    TRANSACTIONS_USER_COL, TRANSACTIONS_ITEM_COL
from azureml.designer.modules.recommendation.dnn.common.entry_param import IntTuple, Boolean
from azureml.designer.modules.recommendation.dnn.common.dataset import TransactionDataset, FeatureDataset
from azureml.designer.modules.recommendation.dnn.common.entry_utils import params_loader
from azureml.designer.modules.recommendation.dnn.wide_and_deep.common.preprocess import preprocess_features, \
    preprocess_transactions
from azureml.designer.modules.recommendation.dnn.common.feature_builder import FeatureBuilder
from azureml.designer.modules.recommendation.dnn.wide_and_deep.common.wide_n_deep_model import WideNDeepModel, \
    WideNDeepModelHyperParams, OptimizerSelection, ActivationFnSelection
from azureml.studio.core.io.model_directory import save_model_to_directory


class TrainWideAndDeepRecommenderModule:
    @staticmethod
    def _validate_features_column_type(dataset: FeatureDataset):
        for col in dataset.columns:
            if dataset.get_column_type(col) == ColumnTypeName.NAN:
                ErrorMapping.throw(InvalidColumnTypeError(col_type=dataset.get_column_type(col),
                                                          col_name=col,
                                                          arg_name=dataset.name))

    @staticmethod
    def _validate_feature_dataset(dataset: FeatureDataset):
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=dataset.column_size,
                                                                       required_column_count=2,
                                                                       arg_name=dataset.name)
        ErrorMapping.verify_number_of_rows_greater_than_or_equal_to(curr_row_count=dataset.row_size,
                                                                    required_row_count=1,
                                                                    arg_name=dataset.name)
        TrainWideAndDeepRecommenderModule._validate_features_column_type(dataset)

    @staticmethod
    def _validate_datasets(transactions: TransactionDataset, user_features: FeatureDataset = None,
                           item_features: FeatureDataset = None):
        ErrorMapping.verify_number_of_columns_equal_to(curr_column_count=transactions.column_size,
                                                       required_column_count=3,
                                                       arg_name=transactions.name)
        ErrorMapping.verify_number_of_rows_greater_than_or_equal_to(curr_row_count=transactions.row_size,
                                                                    required_row_count=1,
                                                                    arg_name=transactions.name)
        ErrorMapping.verify_element_type(type_=transactions.get_column_type(TRANSACTIONS_RATING_COL),
                                         expected_type=ColumnTypeName.NUMERIC,
                                         column_name=transactions.ratings.name,
                                         arg_name=transactions.name)
        if user_features is not None:
            TrainWideAndDeepRecommenderModule._validate_feature_dataset(user_features)
        if item_features is not None:
            TrainWideAndDeepRecommenderModule._validate_feature_dataset(item_features)

    @staticmethod
    def _preprocess(transactions: TransactionDataset, user_features: FeatureDataset, item_features: FeatureDataset):
        # preprocess transactions data
        transactions = preprocess_transactions(transactions)

        # preprocess user features
        user_features = preprocess_features(user_features) if user_features is not None else None
        item_features = preprocess_features(item_features) if item_features is not None else None
        TrainWideAndDeepRecommenderModule._validate_preprocessed_dataset(transactions, user_features=user_features,
                                                                         item_features=item_features)
        return transactions, user_features, item_features

    @staticmethod
    def _validate_preprocessed_dataset(transactions: TransactionDataset, user_features: FeatureDataset,
                                       item_features: FeatureDataset):
        if transactions.row_size <= 0:
            ErrorMapping.throw(
                InvalidDatasetError(dataset1=transactions.name, reason=f"dataset does not have any valid samples"))
        if transactions.df.duplicated(
                subset=transactions.columns[[TRANSACTIONS_USER_COL, TRANSACTIONS_ITEM_COL]]).any():
            ErrorMapping.throw(MoreThanOneRatingError())

        if user_features is not None and any(user_features.df.duplicated(subset=user_features.ids.name)):
            ErrorMapping.throw(DuplicateFeatureDefinitionError())
        if item_features is not None and any(item_features.df.duplicated(subset=item_features.ids.name)):
            ErrorMapping.throw(DuplicateFeatureDefinitionError())

    @staticmethod
    def set_inputs_name(transactions: TransactionDataset, user_features: FeatureDataset = None,
                        item_features: FeatureDataset = None):
        _TRANSACTIONS_NAME = "Training dataset of user-item-rating triples"
        _USER_FEATURES_NAME = "User features"
        _ITEM_FEATURES_NAME = "Item features"
        if transactions is not None:
            transactions.name = _TRANSACTIONS_NAME
        else:
            ErrorMapping.verify_not_null_or_empty(x=transactions, name=_TRANSACTIONS_NAME)
        if user_features is not None:
            user_features.name = _USER_FEATURES_NAME
        if item_features is not None:
            item_features.name = _ITEM_FEATURES_NAME

    @params_loader
    def run(self,
            training_dataset_of_user_item_rating_triples: TransactionDataset,
            user_features: FeatureDataset,
            item_features: FeatureDataset,
            epochs: int,
            batch_size: int,
            wide_part_optimizer: OptimizerSelection,
            wide_optimizer_learning_rate: float,
            crossed_feature_dimension: int,
            deep_part_optimizer: OptimizerSelection,
            deep_optimizer_learning_rate: float,
            user_embedding_dimension: int,
            item_embedding_dimension: int,
            categorical_features_embedding_dimension: int,
            hidden_units: IntTuple,
            activation_function: ActivationFnSelection,
            dropout: float,
            batch_normalization: Boolean,
            trained_wide_and_deep_recommendation_model: str,
            mpi_support: bool = True):
        self.set_inputs_name(training_dataset_of_user_item_rating_triples, user_features=user_features,
                             item_features=item_features)
        self._validate_datasets(training_dataset_of_user_item_rating_triples, user_features=user_features,
                                item_features=item_features)
        self._preprocess(training_dataset_of_user_item_rating_triples, user_features=user_features,
                         item_features=item_features)

        hyper_params = WideNDeepModelHyperParams(epochs=epochs,
                                                 batch_size=batch_size,
                                                 wide_optimizer=wide_part_optimizer,
                                                 wide_lr=wide_optimizer_learning_rate,
                                                 deep_optimizer=deep_part_optimizer,
                                                 deep_lr=deep_optimizer_learning_rate,
                                                 hidden_units=hidden_units,
                                                 activation_fn=activation_function,
                                                 dropout=dropout,
                                                 batch_norm=batch_normalization,
                                                 crossed_dim=crossed_feature_dimension,
                                                 user_dim=user_embedding_dimension,
                                                 item_dim=item_embedding_dimension,
                                                 embed_dim=categorical_features_embedding_dimension)
        user_feature_builder = FeatureBuilder(ids=training_dataset_of_user_item_rating_triples.users,
                                              id_key="User",
                                              features=user_features, feat_key_suffix='user_feature')
        item_feature_builder = FeatureBuilder(ids=training_dataset_of_user_item_rating_triples.items,
                                              id_key="Item",
                                              features=item_features, feat_key_suffix='item_feature')
        model = WideNDeepModel(hyper_params=hyper_params, save_dir=None, user_feature_builder=user_feature_builder,
                               item_feature_builder=item_feature_builder, mpi_support=mpi_support)
        model.train(transactions=training_dataset_of_user_item_rating_triples)
        # trained_wide_and_deep_recommendation_model is trained model output path, and the variable name is
        # defined according to the module spec
        if model.hvd_rank == 0 or not model.mpi_support:
            save_model_to_directory(save_to=trained_wide_and_deep_recommendation_model, model=model)
