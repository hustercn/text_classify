# coding: utf-8
from __future__ import absolute_import

import os
import sys

# fix grpc generated code error
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
from abcft_algorithm_grpc import algorithm_rpc_pb2
from abcft_algorithm_grpc import algorithm_rpc_pb2_grpc
from abcft_algorithm_grpc import algorithm_internal_rpc_pb2
from abcft_algorithm_grpc import algorithm_internal_rpc_pb2_grpc
from abcft_algorithm_grpc import ocr_pb2
from abcft_algorithm_grpc import ocr_pb2_grpc
sys.path = [x for x in sys.path if x != _dir]

from abcft_algorithm_grpc.server import MAX_RECEIV_MESSAGE_SIZE, MAX_SEND_MESSAGE_SIZE
from abcft_algorithm_grpc.server import BaseInternalServer, BaseOcrServer
from abcft_algorithm_grpc.client import BaseInternalClient
from abcft_algorithm_grpc.handler import BaseInternalHandler, BaseOcrHandler
from abcft_algorithm_grpc.reply import Reply
from abcft_algorithm_grpc.ocr_client import OcrClient


__all__ = [
    "algorithm_rpc_pb2",
    "algorithm_rpc_pb2_grpc",
    "algorithm_internal_rpc_pb2",
    "algorithm_internal_rpc_pb2_grpc",
    "ocr_pb2",
    "ocr_pb2_grpc",
    "MAX_RECEIV_MESSAGE_SIZE",
    "MAX_SEND_MESSAGE_SIZE",
    "BaseInternalServer",
    "BaseOcrServer",
    "BaseInternalClient",
    "BaseInternalHandler",
    "BaseOcrHandler",
    "Reply",
    "OcrClient",
]
