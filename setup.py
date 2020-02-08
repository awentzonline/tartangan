import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="tartangan",
    version="0.0.7",
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
            "tartangan_train = tartangan.trainers.iqn:main"
        ]
    },
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'tqdm',
    ]
)
