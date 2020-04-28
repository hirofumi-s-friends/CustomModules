import pandas as pd
import numpy as np
from azureml.studio.core.logger import common_logger
from azureml.studio.internal.error import ErrorMapping, ColumnNotFoundError
from azureml.designer.modules.recommendation.dnn.common.dataset import FeatureDataset
from azureml.studio.core.data_frame_schema import ColumnTypeName
from azureml.designer.modules.recommendation.dnn.common.utils import convert_to_str, convert_to_float


class FeatureMeta:
    """The tool class recorded feature metas.

    Feature meta includes four fields: feature original name, feature type, values used to fill missing elements,
    and the value vocabulary for categorical feature.
    """

    def __init__(self, name, type_, fill, vocab=None):
        self.name = name
        self.type_ = type_
        self.fill = fill
        self.vocab = vocab

    def is_numeric_feature(self):
        return self.vocab is None


class FeatureBuilder:
    """The tool class helps recommenders to build training/test dataset with normalized features.

    The class has three main functionality:
    1. Record (user/item) ids vocab in id_vocab attribute. This attr is usually init during training stage and re-used
    in score stage.
    2. Record features in _features_df and _features_meta attributes. These attrs would be used to build dataset with
    features, which is usually used in hybrid recommendation.
    3. Provide update method to update features for unseen ids during training.
    """

    def __init__(self, ids: pd.Series, id_key: str, features: FeatureDataset = None, feat_key_suffix=None):
        """Init ids and features.

        This function would record ids and features values. And feature metas will also recorded, which can be checked
        in the update method.

        :param ids: represents feature identifiers with pd.Series type
        :param id_key: the key represent id feature
        :param features: FeatureDataset type, where the first column must be feature identifiers, and the remain
        columns are features
        :param feat_key_suffix: when rename features, this string will be append the new names, this param is useful
        when there will be many feature builders and to avoid duplicate feature keys
        """
        ids = convert_to_str(ids).dropna().unique()
        self.id_vocab = pd.Series(ids, name=id_key)

        self._features_df = None
        self._feature_metas = {}
        if features is not None:
            common_logger.info(f"Get {features.features.shape[1]} features")
            features_df = pd.DataFrame({id_key: convert_to_str(features.ids)})
            for idx, name in enumerate(features.features.columns):
                feature = features.features[name]
                column_type = features.get_column_type(name)
                key = f"{name}_{feat_key_suffix}_{idx}" if feat_key_suffix else f"{name}_{idx}"

                normalized_feature = self._normalize_feature(feature=feature, column_type=column_type)
                feature_meta = self._init_feature_meta(feature=normalized_feature, column_type=column_type)
                if feature_meta is not None:
                    features_df[key] = normalized_feature
                    self.feature_metas[key] = feature_meta

            if self.feature_metas:
                self._features_df = features_df
        common_logger.info(f"Create feature metas for {len(self.feature_metas)} features")

    def build(self, ids: pd.Series):
        """Build dataset with normalized features.

        With ids provided, the module checks _features_df attr, and finds corresponding features for those ids.
        If _features_df is None, then only convert ids to string type and return.
        :param ids: the identifiers to built with features
        :return: pd.DataFrame, the dataset built with ids and (optional) features.
        """
        # should ensure returned ids are equal order and length with the input
        common_logger.info("Build ids")
        ids = convert_to_str(ids.dropna())
        features_df = pd.DataFrame({self.id_key: ids}).reset_index(drop=True)

        if self.feature_metas:
            common_logger.info(f"Build {len(self.feature_metas)} features for {self.id_key} ids.")
            features_df = pd.merge(features_df, self._features_df, how='left', on=[self.id_key])
            for key, meta in self.feature_metas.items():
                feature = features_df[key]
                features_df[key] = self._fill_feature_na(feature=feature, feature_meta=meta)

        return features_df

    def update(self, features: FeatureDataset):
        """Update existing features with the new features dataset.

        The method checks the compatibility between existing features and new features dataset. Then update
        unseen identifiers features according to new features. The features for existing identifiers would not be
        updated.
        :param features: FeatureDataset, the new features dataset provided to update the existing features.
        """
        if not self.feature_metas or not features:
            common_logger.info(f"Update feature metas with None features or feature metas")
            return

        self._check_features(features)
        common_logger.info(f"Found {len(self.feature_metas)} features to update")
        feature_names = [meta.name for key, meta in self.feature_metas.items()]
        new_features_df = features.df[[features.ids.name, *feature_names]]
        new_features_df = new_features_df.rename(
            columns=dict((meta.name, key) for key, meta in self.feature_metas.items()))
        new_features_df = new_features_df.rename(columns={features.ids.name: self.id_key})
        new_features_df[self.id_key] = convert_to_str(new_features_df[self.id_key])
        new_features_df = pd.concat([self._features_df, new_features_df], axis=0)
        existed_ids = new_features_df.duplicated(subset=self.id_key, keep='first')
        new_features_df = new_features_df[~existed_ids]

        self._features_df = new_features_df

    def _check_features(self, features: FeatureDataset):
        """Check compatibility between recorded features and the given features.

        The two feature dataset are compatibility if:
        1. The new feature dataset contains all feature names in the old feature dataset
        2. The same feature in two dataset is of same type
        """
        common_logger.info(f"Check features compatibility with existing feature metas")
        for _, feature_meta in self.feature_metas.items():
            name = feature_meta.name
            if name not in features.features:
                ErrorMapping.throw(ColumnNotFoundError(column_id=name, arg_name_missing_column=features.name))
            column_type = features.get_column_type(name)
            if features.get_column_type(name) != feature_meta.type_:
                ErrorMapping.verify_element_type(type_=column_type, expected_type=feature_meta.type_, column_name=name,
                                                 arg_name=features.name)

    @staticmethod
    def _fill_feature_na(feature: pd.Series, feature_meta: FeatureMeta):
        if feature_meta.type_ == ColumnTypeName.NUMERIC:
            feature = feature.replace(to_replace=[np.inf, -np.inf], value=np.nan)
        feature = feature.fillna(feature_meta.fill)

        return feature

    @staticmethod
    def _normalize_feature(feature: pd.Series, column_type):
        if column_type in [ColumnTypeName.CATEGORICAL, ColumnTypeName.BINARY, ColumnTypeName.OBJECT,
                           ColumnTypeName.STRING]:
            new_feature = convert_to_str(column=feature)
        elif column_type == ColumnTypeName.NUMERIC:
            new_feature = convert_to_float(column=feature)
        else:
            new_feature = feature.copy()

        return new_feature

    @staticmethod
    def _init_feature_meta(feature: pd.Series, column_type):
        """Init feature meta for the feature, which is described as pd.Series type.

        The DateTime, TimeSpan and NAN type features would be ignored. And the numeric features with NAN mean
        value is also ignored."""
        if column_type in [ColumnTypeName.DATETIME, ColumnTypeName.TIMESPAN, ColumnTypeName.NAN]:
            common_logger.info(f"Skip {column_type} type feature: {feature.name}")
            feature_meta = None
        elif column_type in [ColumnTypeName.CATEGORICAL, ColumnTypeName.BINARY, ColumnTypeName.OBJECT,
                             ColumnTypeName.STRING]:
            feature_meta = FeatureMeta(name=feature.name, type_=column_type, fill='', vocab=feature.dropna().unique())
        else:
            mean = feature.replace(to_replace=[np.inf, -np.inf], value=np.nan).mean()
            if np.isnan(mean):
                feature_meta = None
                common_logger.info(f"Skip {column_type} feature {feature.name}, the mean value is {mean}")
            else:
                feature_meta = FeatureMeta(name=feature.name, type_=column_type, fill=mean)

        return feature_meta

    @property
    def feature_metas(self):
        return self._feature_metas

    @property
    def id_key(self):
        return self.id_vocab.name
