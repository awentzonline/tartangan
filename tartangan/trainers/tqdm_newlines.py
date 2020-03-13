from tqdm import tqdm
from tqdm._utils import _unicode


class TqdmNewLines(tqdm):
    @staticmethod
    def status_printer(file):
        """
        This is identical to the original implementation with the exception
        of the carriage return being removed and a newline added.
        """
        fp = file
        fp_flush = getattr(fp, 'flush', lambda: None)  # pragma: no cover

        def fp_write(s):
            fp.write(_unicode(s))
            fp_flush()

        last_len = [0]

        def print_status(s):
            len_s = len(s)
            fp_write(s + (' ' * max(last_len[0] - len_s, 0)) + '\n')
            last_len[0] = len_s
        return print_status
