import logging
import sys

import pandas as pd
from pandas import DataFrame

from sys_utils import level, logging_handlers


def progress(msg: str):
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()


def new_line():
    sys.stdout.write("\n")


class SystemLogger:

    @staticmethod
    def logger(name, silent=False) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.progress = progress
        logger.newline = new_line
        if silent:
            return logger
        logger.setLevel(level)
        [logger.addHandler(h) for h in logging_handlers]

        return logger


df_debugger = SystemLogger.logger("df_debugger")


def debug_df(df: DataFrame, prefix: str = None, print_separator: bool = True, max_rows: int = 5):
    with pd.option_context('display.max_columns', None,
                           'display.expand_frame_repr', False):
        if prefix is not None:
            df_debugger.info(prefix)
        df_debugger.info('\n' + str(df.head(max_rows)))
        df_debugger.info(f'Total number of record is %i' % len(df))
    if print_separator:
        df_debugger.info('-' * 80)


if __name__ == '__main__':
    SystemLogger.logger("test").info("this is a test message")
