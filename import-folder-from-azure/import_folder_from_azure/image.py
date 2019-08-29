import os
import fire

from .common import BlobDownloader

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def filter_condition(path):
    _, ext = os.path.splitext(path)
    # The file is not in subdirectory or is an image
    return '/' not in path or ext in IMG_EXTENSIONS


def import_image_folder(account_name, account_key, blob_path, output_folder):
    downloader = BlobDownloader(account_name=account_name, account_key=account_key)
    print('Start:', blob_path)
    downloader.import_folder(blob_path, output_folder, 'AnyDirectory', filter_condition=filter_condition)


if __name__ == '__main__':
    fire.Fire(import_image_folder)
