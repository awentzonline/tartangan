import os
import re

import boto3


s3 = boto3.resource('s3')
s3_client = boto3.client('s3')


def maybe_makedirs(path, exist_ok=True):
    """Don't mkdir if it's a path on S3"""
    if path.startswith('s3://'):
        return
    os.makedirs(path, exist_ok=exist_ok)


def smart_ls(path):
    """Get a list of files from `path`, either S3 or local."""
    if path.startswith('s3://'):
        return _smart_ls_s3(path)
    else:
        return _smart_ls_local(path)


def _smart_ls_s3(path):
    bucket_name, prefix = re.match(r"s3:\/\/(.+?)\/(.+)", path).groups()
    if not prefix.endswith('/'):
        prefix += '/'

    results = []
    paginator = s3_client.get_paginator('list_objects')
    for resp in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/'):
        if not 'CommonPrefixes' in resp:
            break
        for common_prefix in resp['CommonPrefixes']:
            dirname = common_prefix['Prefix'][len(prefix):]  # strip root prefix
            dirname = dirname.rstrip('/')
            results.append(dirname)
    return results


def _smart_ls_local(path):
    if os.path.exists(path):
        return os.listdir(path)
    return []
