import argparse
import pandas as pd
from azureml.studio.core.io.data_frame_directory import save_data_frame_to_directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='inputdir')
    parser.add_argument('--output', default='outputdfd')
    args, _ = parser.parse_known_args()
    df = pd.read_parquet(args.input)
    save_data_frame_to_directory(args.output, data=df, compute_stats_in_visualization=True)
    print(f"Dataframe is saved to {args.output}")
    print(df)
