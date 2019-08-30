import fire
import os
import time
import logging
import urllib.request
from ruamel import yaml

VERSION = '0.0.2'
logger = logging.getLogger('Downloader')
logger.setLevel(logging.DEBUG)
hdl = logging.StreamHandler()
hdl.setFormatter(logging.Formatter('%(asctime)s %(name)-10s %(levelname)-10s %(message)s'))
logger.addHandler(hdl)


def write_meta(model_name, file_name, original_url, yaml_path):
    data = {
        'type': 'ModelDirectory',
        'model_name': model_name,
        'file_name': file_name,
        'original_url': original_url,
    }
    with open(yaml_path, 'w') as fout:
        yaml.round_trip_dump(data, fout)


def reporthook(count, block_size, total_size):
    global start_time
    global last_time
    if count == 0:
        start_time = time.time()
        last_time = time.time()
    # Only show status every 0.5 seconds
    if time.time() - last_time < 0.5:
        return
    last_time = time.time()

    progress_size = count * block_size / 1024 / 1024
    duration = time.time() - start_time
    speed = progress_size / duration
    percent = int(count * block_size * 100 / total_size)
    logger.info(f"{percent}%, {progress_size} MB, {speed} MB/s, {duration} seconds passed")


def import_pretrained_model_from_url(url, output_folder, model_name):
    file_name = url.split('/')[-1]
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Start downloading {file_name} from {url}")
    urllib.request.urlretrieve(url, os.path.join(output_folder, file_name), reporthook)
    logger.info(f"End downloading {file_name} from {url}")

    yaml_file = '_meta.yaml'
    logger.info(f"Start writing yaml file {yaml_file}")
    write_meta(model_name, file_name, url, os.path.join(output_folder, yaml_file))
    logger.info(f"End writing yaml file {yaml_file}")

    logger.info(f"Import pretrained model {model_name} completed.")


if __name__ == '__main__':
    fire.Fire(import_pretrained_model_from_url)
