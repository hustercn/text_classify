# coding: utf-8
from __future__ import absolute_import
from __future__ import unicode_literals

import argparse
import logging

from abcft_algorithm_forecast_extraction.text_classify import classifier
from abcft_algorithm_grpc import cclog
from abcft_algorithm_grpc import algorithm_internal_rpc_pb2
from abcft_algorithm_grpc import BaseInternalHandler
from abcft_algorithm_grpc import BaseInternalServer
log = logging.getLogger(__name__)

__version__ = (0, 1, 0)


def get_version():
    return __version__


class Handler(BaseInternalHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self._versions = [
            {
                "name": "forecast_extraction",
                "version": get_version()
            }
        ]

    def ForecastExtraction(self, request, context):
        log.info("ForecastExtraction")
        # todo : ForecastExtraction
        # result = self.ForecastExtraction.classify(image)
        # for r in rr:
        from abcft_algorithm_forecast_extraction.key_info_extract import \
            parse_forecast

        import json
        r = json.loads(request.fulltext)
        texts = r[0].get('fulltext', [])
        paragraph = r[0].get('paragraphs', [])
        key_info = parse_forecast(texts, paragraph, fmt='html')
        # key_info = json.dumps(key_info, ensure_ascii=False)
        yield algorithm_internal_rpc_pb2.ForecastExtractionReply(
            ok=True,
            result=key_info
        )


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev", help="dev mode, for develop", action="store_true")
    parser.add_argument("-p", "--port", help="RPC Server Port", type=int, default=12400)
    parser.add_argument("--cert_dir", help="cert dir path", default="/opt/algorithm/etc/grpc_certs")
    parser.add_argument("--cert_name", help="cert name", default="grpc-server")
    parser.add_argument("-c", "--cfg", help="config path, json format",
                        default="/opt/algorithm/etc/config/forecast_extraction.rpc.json")
    args = parser.parse_args(args)

    if args.dev:
        cclog.init(level=logging.DEBUG)
        log.info("in dev mode...")
    else:
        cclog.init(level=logging.INFO)
        log.info("in normal mode...")

    log.info("load config from %s", args.cfg)
    # init report text classifier
    classifier.init()
    cert_dir = args.cert_dir
    cert_name = args.cert_name
    endpoint = "0.0.0.0:{}".format(args.port)
    server = BaseInternalServer(
        lambda: Handler(),
        endpoint=endpoint,
        cert_path=cert_dir,
        cert_name=cert_name
    )
    try:
        server.run()
    except Exception as e:
        server.stop()


if __name__ == '__main__':
    exit(main())
