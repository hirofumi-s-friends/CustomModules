import fire

from .common import BlobDownloader


def import_model_folder(account_name, account_key, blob_path, output_folder):
    downloader = BlobDownloader(account_name=account_name, account_key=account_key)
    downloader.import_folder(blob_path, output_folder, 'ModelDirectory')


if __name__ == '__main__':
    fire.Fire(import_model_folder)
