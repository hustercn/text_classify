# coding: utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import os
import codecs
import errno
import hashlib
import logging

_MODELS_DIR_ONLINE_LOCAL = "/opt/algorithm/pb_models/"
_MODELS_DIR_ONLINE_SHARE = "/mnt/share/abcft_pb_models"
_MODELS_DIR_DEV = os.path.expanduser("~/abcft_pb_models")

_MODELS_DIRS = [_MODELS_DIR_ONLINE_LOCAL, _MODELS_DIR_ONLINE_SHARE, _MODELS_DIR_DEV]


class ModelNotFoundError(IOError):
    def __init__(self, name):
        super(IOError, self).__init__(errno.ENOENT, "Model {} not found.".format(name))
        self.model_name = name

class ModelHashMismatchError(IOError):
    def __init__(self, path, expected_hash, real_hash):
        super(IOError, self).__init__(
            errno.ENOENT,
            "Model {} is expected to have MD5 hash {}, but ends up with {}.".format(path, expected_hash, real_hash)
        )
        self.model_path = path
        self.expected_hash = expected_hash
        self.real_hash = real_hash


def get_model(model_name, md5_hash):
    paths = [os.path.join(dirname, model_name) for dirname in _MODELS_DIRS]
    model_path = None
    for p in paths:
        if os.path.exists(p):
            model_path = p
            break
    if not model_path:
        raise ModelNotFoundError(model_name)

    with codecs.open(model_path, "rb") as f:
        data = f.read()
    m = hashlib.md5(data).hexdigest()
    if m != md5_hash.lower():
        raise ModelHashMismatchError(model_path, md5_hash, m)

    logging.info("successfully load model %s with MD5 hash %s", model_path, md5_hash)
    return model_path
