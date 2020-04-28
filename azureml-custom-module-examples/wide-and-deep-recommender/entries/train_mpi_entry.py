# flake8: noqa: E402
import azureml
import importlib

# This is a workaround to make sure azureml in local directory could be loaded.
importlib.reload(azureml)
print("Path of azureml")
print('\n'.join(azureml.__path__))

from azureml.designer.modules.recommendation.dnn.wide_and_deep.train.train_wide_and_deep_recommender import \
    TrainWideAndDeepRecommenderModule
from azureml.designer.modules.recommendation.dnn.common.entry_utils import build_cli_args


def main():
    kwargs = build_cli_args(TrainWideAndDeepRecommenderModule().run)
    TrainWideAndDeepRecommenderModule().run(**kwargs)


if __name__ == "__main__":
    main()
