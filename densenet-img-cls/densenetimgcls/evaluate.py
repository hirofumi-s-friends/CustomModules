from torchvision import datasets
from azureml.studio.common.datatypes import DataTypes
from azureml.studio.common.datatable.data_table import DataTable
from azureml.studio.modulehost.handler.port_io_handler import OutputHandler
from .utils import get_transform, evaluate, load_model, logger
import os
import fire
import torch
import pandas as pd
import json


def cal_metric(model, test_set, batch_size):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    _, _, test_error, (target_list, pred_top1_list) = evaluate(model=model, loader=test_loader, is_test=True)
    logger.info(f'Final test error: {test_error:.4f}, accuracy: {1 - test_error:.4f}')
    test_filename = ['/'.join(i[0].split('/')[-2:]) for i in test_set.samples]
    pred_df = pd.DataFrame({'filename': test_filename, 'target': target_list, 'pred_top1': pred_top1_list})
    return pd.DataFrame({'Top 1 accuracy': [float(f"{1 - test_error:.5f}")]}), pred_df


def entrance(model_path='saved_model', model_type='densenet201', memory_efficient=False, num_classes=3,
             data_path='', batch_size=128, metric_save_path='metric_save', pred_save_path='pred_save'):
    with open(os.path.join(model_path, 'index_to_label.json')) as f:
        classes = json.load(f)
    os.makedirs(metric_save_path, exist_ok=True)
    os.makedirs(pred_save_path, exist_ok=True)
    model = load_model(model_path, model_type, memory_efficient, num_classes=num_classes)
    _, test_transforms = get_transform()
    test_set = datasets.ImageFolder(data_path, transform=test_transforms)
    metric_df, pred_df = cal_metric(model, test_set, batch_size)
    # Map to class names
    pred_df['target'] = pred_df['target'].map(lambda x: classes[x])
    pred_df['pred_top1'] = pred_df['pred_top1'].map(lambda x: classes[x])
    # Save as dataframe
    OutputHandler.handle_output(
        data=DataTable(metric_df),
        file_path=metric_save_path,
        file_name='data.dataset.parquet',
        data_type=DataTypes.DATASET,
    )
    OutputHandler.handle_output(
        data=DataTable(pred_df),
        file_path=pred_save_path,
        file_name='data.dataset.parquet',
        data_type=DataTypes.DATASET,
    )
    logger.info('This experiment has been completed.')


if __name__ == '__main__':
    fire.Fire(entrance)
