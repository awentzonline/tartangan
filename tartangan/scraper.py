"""
Gently warm the caches of tartanregister.gov.uk.
"""
import argparse
import json
import os
import random
import shutil
import time

import requests

from tartangan.utils.fs import maybe_makedirs


def scrape_tartans(args):
    """
    Slowly, serially download images so as not to wear out our welcome.
    """
    maybe_makedirs(args.output_path, exist_ok=True)
    print('Scraping tartans')
    # prepare list of ids to scrape, possibly resume work
    ids_to_scrape = load_state(args.state)
    if ids_to_scrape is None:
        ids_to_scrape = list(range(1, args.max_id))
        random.shuffle(ids_to_scrape)
        errors = []
    else:
        errors = load_state(args.errors) or []
    num_processed = 0
    while ids_to_scrape:
        page_id = ids_to_scrape.pop()
        url = args.url_template.format(
            page_id=page_id, width=args.size, height=args.size
        )
        print(url)
        filename = os.path.join(args.output_path, f'{page_id}.jpg')
        error = download_image_url(url, filename)
        if error:
            errors.append([page_id, error])
            print(error)
        num_processed += 1
        if num_processed % args.save_state_freq == 0:
            save_state(ids_to_scrape, args.state)
            save_state(errors, args.errors)
        # we're decent people who just want some images
        time.sleep(args.sleep)


def load_state(filename):
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as infile:
        state = json.load(infile)
    return state


def save_state(state, filename):
    with open(filename, 'w') as outfile:
        json.dump(state, outfile)


def download_image_url(url, output_filename):
    """
    https://stackoverflow.com/questions/13137817/how-to-download-image-using-requests
    """
    res = requests.get(url, stream=True)
    if res.status_code == 200 and res.headers['content-type'] in ('image/jpeg',):
        with open(output_filename, 'wb') as outfile:
            res.raw.decode_content = True
            shutil.copyfileobj(res.raw, outfile)
    else:
        content = res.content.decode('utf-8')
        if 'The tartan details provided cannot be converted' in content:
            return [res.status_code, 'Tartan not found']
        return [res.status_code, content]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output_path')
    p.add_argument(
        '--url-template',
        default='https://www.tartanregister.gov.uk/imageCreation?ref={page_id}&width={width}&height={height}'
    )
    p.add_argument('--state', default='scraper_state.json')
    p.add_argument('--errors', default='scraper_errors.json')
    p.add_argument('--size', type=int, default=750)
    p.add_argument('--sleep', type=float, default=1.)
    p.add_argument('--save-state-freq', type=int, default=5)
    p.add_argument('--max-id', type=int, default=12698)
    args = p.parse_args()
    scrape_tartans(args)


if __name__ == '__main__':
    main()
