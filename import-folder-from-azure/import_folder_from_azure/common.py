import os
import logging

from ruamel import yaml
from azure.storage.blob import BlockBlobService

logger = logging.getLogger('Downloader')
logger.setLevel(logging.DEBUG)
hdl = logging.StreamHandler()
hdl.setFormatter(logging.Formatter('%(asctime)s %(name)-10s %(levelname)-10s %(message)s'))
logger.addHandler(hdl)


class BlobDownloader:
    META_FILE = '_meta.yaml'
    _URL_DELIMITER = '/'

    def __init__(self, account_name, account_key):
        self.blob_service = BlockBlobService(account_name=account_name, account_key=account_key)
        self.logger = logger

    def download_files_from_blob(self, container, blob_paths, prefix_len, output_folder):
        files = []
        for blob_path in blob_paths:
            file_name = blob_path[prefix_len:]

            folder = os.path.dirname(os.path.join(output_folder, file_name))
            os.makedirs(folder, exist_ok=True)

            self.logger.info(f"Start downloading file: {file_name}")
            self.blob_service.get_blob_to_path(
                container_name=container,
                blob_name=blob_path,
                file_path=os.path.join(output_folder, file_name),
            )
            self.logger.info(f"End downloading file: {file_name}")
            files.append(file_name)
        if len(files) == 0:
            raise Exception("File not found in path")
        self.logger.info(f"{len(files)} files downloaded")

    def import_folder(self, blob_path, output_folder, folder_type='GenericFolder'):
        parts = blob_path.strip(self._URL_DELIMITER).split(self._URL_DELIMITER)
        container = parts[0]
        blob_path = self._URL_DELIMITER.join(parts[1:]) + self._URL_DELIMITER
        if blob_path == '/':
            blob_path = ''
        self.logger.info(f"Blob path parsed: container={container}, path={blob_path}")
        blob_paths = self.blob_service.list_blob_names(container, prefix=blob_path)
        self.download_files_from_blob(container, blob_paths, len(blob_path), output_folder)
        create_if_not_exist = folder_type != 'GenericFolder'
        self.ensure_meta(output_folder, folder_type, create_if_not_exist)

    def ensure_meta(self, output_folder, folder_type, create_if_not_exist=True):
        meta_file_path = os.path.join(output_folder, self.META_FILE)
        if os.path.exists(meta_file_path):
            self._check_meta(meta_file_path, folder_type)
        elif create_if_not_exist:
            self._generate_meta(meta_file_path, folder_type)

    @staticmethod
    def _check_meta(meta_file_path, folder_type):
        try:
            with open(meta_file_path) as fin:
                data = yaml.safe_load(fin)
            if data['type'] != folder_type:
                raise Exception(f"Invalid folder_type in meta, expected: {folder_type}, got {data['type']}")
        except BaseException as e:
            raise Exception(f"Invalid meta file") from e

    @staticmethod
    def _generate_meta(meta_file_path, folder_type):
        data = {'type': folder_type}
        with open(meta_file_path, 'w') as fout:
            yaml.round_trip_dump(data, fout)
        logger.info(f"Meta file created: {meta_file_path}")
