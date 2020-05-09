import fire
import os
import sys
import time
import urllib.request
from ruamel import yaml


def write_meta(file_name, original_url, yaml_path):
    data = {'type': 'ModelFolder', 'pretrained_file': file_name, 'original_url': original_url}
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
    print(f"{percent}%, {progress_size} MB, {speed} MB/s, {duration} seconds passed")
    sys.stdout.flush()


def import_pretrained_model_from_url(url, output_folder, file_name=""):
    file_name = file_name.strip()
    if file_name == '':
        file_name = url.split('/')[-1]
    os.makedirs(output_folder, exist_ok=True)
    print(f"Start downloading {file_name} from {url}")
    urllib.request.urlretrieve(url, os.path.join(output_folder, file_name), reporthook)
    print(f"End downloading {file_name} from {url}")

    yaml_file = '_meta.yml'
    print(f"Start writing yaml file {yaml_file}")
    write_meta(file_name, url, os.path.join(output_folder, yaml_file))
    print(f"End writing yaml file {yaml_file}")


if __name__ == '__main__':
    fire.Fire(import_pretrained_model_from_url)
