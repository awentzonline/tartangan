from tartangan.trainers.components.base import TrainerComponent


class FileBasedMetricsComponent(TrainerComponent):
    @classmethod
    def add_args_to_parser(self, parser):
        parser.add_argument('--metrics-path', default=None,
                            help='Where to output a file containing run metrics')
