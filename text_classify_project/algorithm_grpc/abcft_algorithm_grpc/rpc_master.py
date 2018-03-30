# coding: utf-8
from __future__ import absolute_import
from __future__ import unicode_literals

import argparse
import logging

from abcft_algorithm_grpc import algorithm_internal_rpc_pb2
from abcft_algorithm_grpc import BaseInternalHandler
from abcft_algorithm_grpc import BaseInternalServer
from abcft_algorithm_grpc.client import BaseInternalClient
from abcft_algorithm_grpc.utils import get_available_port, get_host_ip
from abcft_algorithm_grpc.zkutils import ServiceAgent


log = logging.getLogger(__name__)


class Handler(BaseInternalHandler):
    def __init__(self, proxies):
        super(Handler, self).__init__()
        self._clients = {
            p["name"]: BaseInternalClient(host=p["host"], port=p["port"]) for p in proxies
        }
        for name in self._clients:
            v = self._get_client(name).GetVersions()
            self._versions.extend(v.versions)

    def _get_client(self, name):
        return self._clients.get(name)

    def GetVersions(self, request, context):
        reply = algorithm_internal_rpc_pb2.VersionsReply()
        for name in self._clients:
            v = self._get_client(name).GetVersions()
            reply.versions.extend(v.versions)
        return reply

    def ImageDetect(self, request, context):
        log.info("ImageDetect")
        client = self._get_client("image_detect")
        if not client:
            raise RuntimeError("no image_detect services running")
        rr = client.stub.ImageDetect(request)
        for r in rr:
            yield r

    def ImageClassify(self, request, context):
        log.info("ImageClassify")
        client = self._get_client("image_classify")
        if not client:
            raise RuntimeError("no image_classify services running")
        rr = client.stub.ImageClassify(request)
        for r in rr:
            yield r

    def TitleClassify(self, request, context):
        log.info("TitleClassify")
        client = self._get_client("title_classify")
        if not client:
            raise RuntimeError("no title_classify services running")
        rr = client.stub.TitleClassify(request)
        for r in rr:
            yield r

    def BitmapTableParse(self, request, context):
        log.info("BitmapTableParse")
        client = self._get_client("bitmap_table_parse")
        if not client:
            raise RuntimeError("no bitmap_table_parse services running")
        rr = client.stub.BitmapTableParse(request)
        for r in rr:
            yield r


def get_proxies_from_args(args):
    proxies = []
    if args.proxy_image_detect:
        h, p = args.proxy_image_detect.split(":")
        proxies.append({
            "name": "image_detect",
            "host": h,
            "port": int(p),
        })
    if args.proxy_title_classify:
        h, p = args.proxy_title_classify.split(":")
        proxies.append({
            "name": "title_classify",
            "host": h,
            "port": int(p),
        })
    if args.proxy_image_classify:
        h, p = args.proxy_image_classify.split(":")
        proxies.append({
            "name": "image_classify",
            "host": h,
            "port": int(p),
        })
    if args.proxy_bitmap_table_parse:
        h, p = args.proxy_bitmap_table_parse.split(":")
        proxies.append({
            "name": "bitmap_table_parse",
            "host": h,
            "port": int(p),
        })
    return proxies


def main(args=None):
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev", help="dev mode, for develop, otherwise ignore all arguments", action="store_true")
    parser.add_argument("-p", "--port", help="RPC Server Port", type=int, default=12300)
    parser.add_argument("--cert_dir", help="cert dir path", default="/opt/algorithm/etc/grpc_certs")
    parser.add_argument("--cert_name", help="cert name", default="grpc-server")
    parser.add_argument("--proxy_image_detect", help="proxy_image_detect", default=None)
    parser.add_argument("--proxy_title_classify", help="proxy_title_classify", default=None)
    parser.add_argument("--proxy_image_classify", help="proxy_image_classify", default=None)
    parser.add_argument("--proxy_bitmap_table_parse", help="proxy_bitmap_table_parse", default=None)
    args = parser.parse_args(args)

    if args.dev:
        log.info("in dev mode...")
        sa = None
        port = args.port
        cert_dir = args.cert_dir
        cert_name = args.cert_name
        proxies = get_proxies_from_args(args)
    else:
        log.info("in distribute mode...")
        sa = ServiceAgent("internal_rpc_master")
        # for now, hard-coded
        cert_dir = "/opt/algorithm/etc/grpc_certs"
        cert_name = "grpc-server"
        # get configure from zookeeper
        cfg = sa.get_configure()
        port = int(cfg["base_port"]) + 1
        proxies = sa.get_all_proxies()

    host = get_host_ip()
    port = get_available_port(port, 10)
    server = BaseInternalServer(
        lambda : Handler(proxies),
        host=host,
        port=port,
        cert_path=cert_dir,
        cert_name=cert_name
    )
    endpoint = "{}:{}".format(host, port)
    try:
        if sa: sa.start_serve(endpoint)
        server.run()
    except Exception as e:
        if sa: sa.stop_serve()
        server.stop()


if __name__ == '__main__':
    exit(main())
