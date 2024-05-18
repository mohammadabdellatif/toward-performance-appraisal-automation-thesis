import logging
import sys

LOG_FORMAT = '%(asctime)s %(levelname)-8s %(message)s'
LOGGER_DT_FORMAT = '%Y-%m-%d %H:%M:%S'

logging_handlers = []
level = logging.INFO


def configure_console_handler():
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOGGER_DT_FORMAT)
    stream_handler.setFormatter(formatter)
    return stream_handler


logging_handlers.append(configure_console_handler())
