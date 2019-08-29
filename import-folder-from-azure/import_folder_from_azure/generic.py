import fire

from .common import BlobDownloader


def import_generic_folder(account_name, account_key, blob_path, output_folder):
    downloader = BlobDownloader(account_name=account_name, account_key=account_key)
    print('Start:', blob_path)
    downloader.import_folder(blob_path, output_folder, 'AnyDirectory')


if __name__ == '__main__':
    fire.Fire(import_generic_folder)
