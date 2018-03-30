# coding: utf-8
from __future__ import unicode_literals
from __future__ import print_function

import sys
import logging
import logging.handlers

NOTSET = logging.NOTSET
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

CCLOG_FORMAT = "%(asctime)s [%(process)d] [%(levelname)s] [%(name)s] [%(funcName)s]: %(message)s"


class CCFormatter(logging.Formatter):
    COLORS = {
        DEBUG: "\x1b[37m",
        INFO: "\x1b[32m",
        WARNING: "\x1b[33m",
        ERROR: "\x1b[31m",
        CRITICAL: "\x1b[35m",
        "default": "\x1b[0m"
    }

    def __int__(self, fmt=None):
        super(CCFormatter, self).__init__(fmt)

    def format(self, record):
        result = logging.Formatter.format(self, record)
        nocolor = CCFormatter.COLORS["default"]
        color = CCFormatter.COLORS.get(record.levelno, nocolor)
        return color + result + nocolor


def init(**kwargs):
    logfile = kwargs.pop("logfile", sys.stderr)
    filesize = kwargs.pop("filesize", 10)
    filecount = kwargs.pop("filecount", 20)
    fmt = kwargs.pop("format", CCLOG_FORMAT)
    level = kwargs.pop("level", DEBUG)
    disable = kwargs.pop("disable", [])

    if len(kwargs) > 0:
        k, _ = kwargs.popitem()
        raise TypeError("invalid keyword argument '{}'".format(k))

    formatter = CCFormatter(fmt)
    if logfile == sys.stderr or logfile == sys.stdout:
        handler = logging.StreamHandler(logfile)
    else:
        maxBytes = filesize * 1024 * 1024
        handler = logging.handlers.RotatingFileHandler(
            logfile,
            maxBytes=maxBytes,
            backupCount=filecount,
            encoding="utf-8"
        )
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    logging.root.setLevel(level)
    for dis in disable:
        logging.getLogger(dis).propagate = False


if __name__ == '__main__':
    init()
    log = logging.getLogger("cclog")
    log.debug("debug")
    log.info("info")
    log.warning("warning")
    log.error("error")
    log.critical("critical")
