import os
import fire
import base64
import pandas as pd

from azureml.studio.common.datatypes import DataTypes
from azureml.studio.common.datatable.data_table import DataTable
from azureml.studio.modulehost.handler.port_io_handler import OutputHandler

VERSION = '0.0.7'
IMG_EXTS = {'.jfif', '.png', '.jpg', '.jpeg'}


def img2base64(f):
    with open(f, 'rb') as fin:
        b64data = base64.b64encode(fin.read()).decode('ascii')
        return f'data:image/png;base64,{b64data}'


def image_to_df(image_path, output_path):
    imgs = []
    encoder = img2base64
    for f in os.listdir(image_path):
        _, ext = os.path.splitext(f)
        if ext not in IMG_EXTS:
            continue
        print(f"Loading image {f}")
        imgs.append(encoder(os.path.join(image_path, f)))

    if not imgs:
        raise FileNotFoundError(f"No valid image file in path: {image_path}")

    df = pd.DataFrame({'image_string': imgs})
    OutputHandler.handle_output(
        data=DataTable(df),
        file_path=output_path,
        file_name='data.dataset.parquet',
        data_type=DataTypes.DATASET,
    )
    print(f"DataFrame dumped: {df}")


if __name__ == '__main__':
    print(f"Image to DataFrame version: {VERSION}")
    fire.Fire(image_to_df)
