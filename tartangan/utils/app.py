import argparse


class App:
    app_name = 'base app'

    def __init__(self, args):
        self.args = args

    def run(self):
        pass

    @classmethod
    def create_from_cli(cls):
        args = cls.parse_cli_args()
        obj = cls(args)
        return obj

    @classmethod
    def parse_cli_args(cls):
        p = argparse.ArgumentParser(
            description=cls.app_name, fromfile_prefix_chars='@'
        )
        cls.add_args_to_parser(p)
        return p.parse_args()

    @classmethod
    def add_args_to_parser(cls, p):
        pass
