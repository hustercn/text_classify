# coding: utf-8
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import codecs
import socket
import functools
import json
import logging

import requests


def load_certificate(path, name):
    cert_path = os.path.join(path, name)
    try:
        with codecs.open(cert_path, "rb") as fp:
            return fp.read()
    except IOError:
        raise IOError("can not load certificate {}".format(cert_path))


def get_file_data(url):
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    return r.content


def get_available_port(first, count):
    port = None
    for p in range(first, first + count):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("localhost", p))
            port = p
        except socket.error:
            pass
        finally:
            s.close()
        if port:
            break
    return port


def get_full_hostname():
    return socket.getfqdn()


def get_hostname():
    return socket.gethostname()


def get_host_ip():
    return socket.gethostbyname(socket.gethostname())


def safe_run(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(e)
            return None
    return wrapper


def load_config(cfg_path):
    if cfg_path.startswith("http://") or cfg_path.startswith("https://"):
        r = requests.get(cfg_path)
        return r.json()
    else:
        return json.load(codecs.open(cfg_path, "r", encoding="utf-8"))
