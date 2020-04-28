# flake8: noqa: E402
# This is a workaround to make sure azureml in local directory could be loaded.
import azureml
import importlib
importlib.reload(azureml)
print(azureml.__path__)

import os
import argparse
from pathlib import Path
import pandas as pd

from azureml.designer.modules.recommendation.dnn.wide_and_deep.score. \
    score_wide_and_deep_recommender import ScoreWideAndDeepRecommenderModule
from azureml.designer.modules.recommendation.dnn.wide_and_deep.common.wide_n_deep_model import WideNDeepModel
from azureml.designer.modules.recommendation.dnn.common.entry_utils import build_cli_args
from azureml.studio.core.io.data_frame_directory import DataFrameDirectory
from azureml.studio.core.io.model_directory import ModelDirectory
from azureml.studio.core.utils.fileutils import iter_files


def init():
    global kwargs
    kwargs = build_cli_args(ScoreWideAndDeepRecommenderModule().run)
    model_key = 'trained_wide_and_deep_recommendation_model'
    input_dir = kwargs[model_key]
    for f in iter_files(input_dir):
        print(f)
    kwargs[model_key] = ModelDirectory.load_instance(load_from_dir=input_dir, model_class=WideNDeepModel)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputdir')
    args, _ = parser.parse_known_args()

    output_dir = Path(args.output)
    # We dump a mock DataFrameDirectory to let the downstream module could read the parquets as a DataFrameDirectory
    mock_result_dfd = DataFrameDirectory.create(
        data=pd.DataFrame(),
        compute_visualization=False,
        compute_schema_if_not_exist=False,
    )
    mock_result_dfd.dump(output_dir)
    global output_df_dir
    output_df_dir = output_dir / mock_result_dfd.meta.data
    print("Output dataframe dir:", output_df_dir)
    output_df_dir.mkdir(exist_ok=True)
    print()


def run(files):
    results = []
    for f in files:
        df = pd.read_parquet(f)
        dfd = DataFrameDirectory.create(data=df)
        kwargs['dataset_to_score'] = dfd
        scored_dfd, = ScoreWideAndDeepRecommenderModule().run(**kwargs)
        fname = Path(f).name
        output_path = output_df_dir / fname
        scored_dfd.data.to_parquet(output_path)
        print(f"Score finished, shape of df = {df.shape}, columns={df.columns.tolist()} output path={output_path}")
        print()
        results.append(str(output_path))
    return results


if __name__ == '__main__':
    init()
    basedir = 'inputdir'
    files = [os.path.join(basedir, f) for f in os.listdir(basedir)]
    print(files)
    print(run(files))
    dfd = DataFrameDirectory.load('outputdir')
    print(dfd.data)
