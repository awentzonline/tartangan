import os


def maybe_makedirs(path, exist_ok=True):
    """Don't mkdir if it's a path on S3"""
    if path.startswith('s3://'):
        return
    os.makedirs(path, exist_ok=exist_ok)
