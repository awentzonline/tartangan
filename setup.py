import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="tartangan",
    version="0.4.0",
    author="Adam Wentz",
    author_email="adam@adamwentz.com",
    description="Model tartan patterns with a GAN.",
    long_description=read("README.md"),
    license="MIT",
    url="https://github.com/awentzonline/tartangan",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "tartangan_scrape = tartangan.scraper:main",
            "tartangan_train_cnn = tartangan.trainers.cnn:main",
            "tartangan_train_iqn = tartangan.trainers.iqn:main",
        ]
    },
    install_requires=[
        'numpy',
        'smart_open',
        'torch',
        'torchvision',
        'tqdm',
    ]
)
