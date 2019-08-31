import os
import json


def smt_fake_file(save_path):
    # Dump data_type.json as a work around until SMT deploys
    dct = {
        'Id': 'Dataset',
        'Name': 'Dataset .NET file',
        'ShortName': 'Dataset',
        'Description': 'A serialized DataTable supporting partial reads and writes',
        'IsDirectory': False,
        'Owner': 'Microsoft Corporation',
        'FileExtension': 'dataset.parquet',
        'ContentType': 'application/octet-stream',
        'AllowUpload': False,
        'AllowPromotion': True,
        'AllowModelPromotion': False,
        'AuxiliaryFileExtension': None,
        'AuxiliaryContentType': None
    }
    with open(os.path.join(save_path, 'data_type.json'), 'w') as f:
        json.dump(dct, f)


def smt_fake_model(save_path):
    # Dump data_type.json as a work around until SMT deploys
    dct = {
        'Id': 'ILearnerDotNet',
        'Name': 'ILearner .NET file',
        'ShortName': 'Model',
        'Description': 'A .NET serialized ILearner',
        'IsDirectory': False,
        'Owner': 'Microsoft Corporation',
        'FileExtension': 'ilearner',
        'ContentType': 'application/octet-stream',
        'AllowUpload': False,
        'AllowPromotion': False,
        'AllowModelPromotion': True,
        'AuxiliaryFileExtension': None,
        'AuxiliaryContentType': None
    }
    with open(os.path.join(save_path, 'data_type.json'), 'w') as f:
        json.dump(dct, f)
    # Dump data.ilearner as a work around until data type design
    visualization = os.path.join(save_path, 'data.ilearner')
    with open(visualization, 'w') as file:
        file.writelines('{}')
