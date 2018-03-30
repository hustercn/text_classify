# coding: utf-8
from __future__ import absolute_import
from __future__ import unicode_literals

import time
import logging
from concurrent import futures
try:
    from multiprocessing import cpu_count as default_thread_count
except ImportError:
    # some platforms don't have multiprocessing
    default_thread_count = lambda : None

import grpc

from abcft_algorithm_grpc import algorithm_rpc_pb2_grpc
from abcft_algorithm_grpc import algorithm_internal_rpc_pb2_grpc
from abcft_algorithm_grpc import ocr_pb2_grpc
from abcft_algorithm_grpc.utils import load_certificate


log = logging.getLogger(__name__)

MAX_RECEIV_MESSAGE_SIZE = 32 << 20
MAX_SEND_MESSAGE_SIZE = 32 << 20

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 12300
DEFAULT_CERT_PATH = "/opt/algorithm/etc/grpc_certs"
DEFAULT_CERT_NAME = "grpc-server"
DEFAULT_CERT_CA_NAME = "rootCA"


class BaseServer(object):
    def __init__(self, handler_cls, server_register,
                 host=DEFAULT_HOST, port=DEFAULT_PORT, endpoint=None,
                 cert_path=DEFAULT_CERT_PATH, cert_name=DEFAULT_CERT_NAME, cert_ca_name=DEFAULT_CERT_CA_NAME,
                 threads=None, thread_pool=None):

        if threads is None:
            threads = default_thread_count() or 4

        self._threads = threads
        if thread_pool is None:
            thread_pool = futures.ThreadPoolExecutor()

        self._server = grpc.server(
            thread_pool,
            options=(
                ('grpc.max_receive_message_length', MAX_RECEIV_MESSAGE_SIZE),
                ('grpc.max_send_message_length', MAX_SEND_MESSAGE_SIZE),
            )
        )

        server_register(handler_cls(), self._server)

        if endpoint:
            self._endpoint = endpoint
        else:
            self._endpoint = "{}:{}".format(host, port)
        log.info("Listen endpoint at %s", self._endpoint)

        private_key_cert = load_certificate(cert_path, "{}.pem".format(cert_name))
        cert_chain = load_certificate(cert_path, "{}.crt".format(cert_name))
        root_cert = load_certificate(cert_path, "{}.pem".format(cert_ca_name))


        credentials = grpc.ssl_server_credentials([(private_key_cert, cert_chain)], root_cert, True)
        self._server.add_secure_port(self._endpoint, credentials)


    @property
    def endpoint(self):
        return self._endpoint

    @property
    def threads(self):
        return self._threads

    def run(self):
        self._server.start()
        try:
            while True:
                time.sleep(3600 * 24)
        except KeyboardInterrupt:
            self._server.stop(0)

    def stop(self):
        self._server.stop(0)


class BasePublicServer(BaseServer):
    def __init__(self, handler_cls, endpoint=None,
                 cert_path=DEFAULT_CERT_PATH, cert_name=DEFAULT_CERT_NAME, cert_ca_name=DEFAULT_CERT_CA_NAME,
                 threads=None, thread_pool=None):
        super(BasePublicServer, self).__init__(
            handler_cls, algorithm_rpc_pb2_grpc.add_AlgorithmServiceServicer_to_server, endpoint=endpoint,
            cert_path=cert_path, cert_name=cert_name, cert_ca_name=cert_ca_name,
            threads=threads, thread_pool=thread_pool
        )


class BaseInternalServer(BaseServer):
    def __init__(self, handler_cls, host=DEFAULT_HOST, port=DEFAULT_PORT, endpoint=None,
                 cert_path=DEFAULT_CERT_PATH, cert_name=DEFAULT_CERT_NAME, cert_ca_name=DEFAULT_CERT_CA_NAME,
                 threads=None, thread_pool=None):
        super(BaseInternalServer, self).__init__(
            handler_cls, algorithm_internal_rpc_pb2_grpc.add_AlgorithmInternalServiceServicer_to_server,
            host=host, port=port, endpoint=endpoint,
            cert_path=cert_path, cert_name=cert_name, cert_ca_name=cert_ca_name,
            threads=threads, thread_pool=thread_pool
        )


class BaseOcrServer(BaseServer):
    def __init__(self, handler_cls, endpoint=None,
                 cert_path=DEFAULT_CERT_PATH, cert_name="ocr-server", cert_ca_name=DEFAULT_CERT_CA_NAME,
                 threads=None, thread_pool=None):
        super(BaseOcrServer, self).__init__(
            handler_cls, ocr_pb2_grpc.add_OcrServiceServicer_to_server, endpoint=endpoint,
            cert_path=cert_path, cert_name=cert_name, cert_ca_name=cert_ca_name,
            threads=threads, thread_pool=thread_pool
        )
