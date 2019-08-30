from setuptools import setup
from import_model_folder.pretrained_from_url import VERSION

setup(
    name="azureml-custom-module-import-model-folder",
    version=VERSION,
    description="A custom module for importing model folder.",
    packages=["import_model_folder"],
    author="Heyi Tang",
    license="MIT",
    include_package_data=True,
)
