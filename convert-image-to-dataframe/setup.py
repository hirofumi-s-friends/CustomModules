from setuptools import setup


setup(
    name="azureml-custom-module-convert-image-to-dataframe",
    version="0.0.6",
    description="A custom module for converting an image directory to dataframe directory.",
    packages=["convert_image_to_dataframe"],
    author="Heyi Tang",
    license="MIT",
    include_package_data=True,
)
