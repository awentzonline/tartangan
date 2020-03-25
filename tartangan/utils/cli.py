import sys

import smart_open


def save_cli_arguments(filename, fromfile_prefix='@'):
    """
    Save the commandline arguments to a file suitable for use
    with argparse.ArgumentParser file input.

    If the only argument is a filename, make a copy of that file.
    """
    args = sys.argv[1:]
    # if it's using file input copy the args out of it
    if args and args[0].startswith(fromfile_prefix):
        input_filename = args[0][1:]  # strip @ from beginning of filename
        with smart_open.open(input_filename, 'r') as infile:
            args = infile.readlines()
            args = [line.strip() for line in args]

    with smart_open.open(filename, 'w') as outfile:
        outfile.write('\n'.join(args))
