# coding: utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

from six import text_type
import logging

from abcft_algorithm_grpc import ocr_pb2
from abcft_algorithm_grpc.client import BaseClient
from abcft_algorithm_grpc.client import DEFAULT_CERT_PATH, \
    DEFAULT_CLIENT_CERT_NAME, DEFAULT_CERT_CA_NAME

log = logging.getLogger(__name__)


class OcrClient(BaseClient):
    def __init__(self, endpoint=None, cert_path=DEFAULT_CERT_PATH,
                 cert_name=DEFAULT_CLIENT_CERT_NAME,
                 cert_ca_name=DEFAULT_CERT_CA_NAME):
        super(OcrClient, self).__init__(
            ocr_pb2.OcrServiceStub,
            service_name="ocr", endpoint=endpoint,
            cert_path=cert_path, cert_name=cert_name, cert_ca_name=cert_ca_name
        )

    def detectElement(self, url_or_data, need_denoise=False,
                      need_denoise_image=False, need_parse_title=False):
        req = ocr_pb2.OcrRequest()
        if isinstance(url_or_data, text_type):
            req.url = url_or_data
        else:
            req.data = url_or_data
        req.need_denoise = need_denoise
        req.need_denoise_image = need_denoise_image
        req.need_parse_title = need_parse_title
        r = self.stub.detectElement(req)
        if r.status.success:
            return {
                "elements": [{"type": e.type,
                              "bbox": ((e.bbox.center_x, e.bbox.center_y),
                                       (e.bbox.width, e.bbox.height),
                                       e.bbox.rotate),
                              "text": e.text} for e in r.elements],
                "image": r.image
            }
        else:
            log.error("ocr rpc detectElement error: %s => %s",
                      r.status.error_code, r.status.error_msg)
            return None

    def predictText(self, url_or_data):
        if isinstance(url_or_data, text_type):
            req = ocr_pb2.OcrRequest(url=url_or_data)
        else:
            req = ocr_pb2.OcrRequest(data=url_or_data)
        r = self.stub.predictText(req)
        if r.status.success:
            return r.texts[0]
        else:
            log.error("ocr rpc predictText error: %s => %s",
                      r.status.error_code, r.status.error_msg)
            return None

    def batchPredictText(self, url_or_datas):
        batch_req = ocr_pb2.BatchOcrRequest()
        reqs = []
        for url_or_data in url_or_datas:
            if isinstance(url_or_data, text_type):
                req = ocr_pb2.OcrRequest(url=url_or_data)
            else:
                req = ocr_pb2.OcrRequest(data=url_or_data)
            reqs.append(req)
        batch_req.images.extend(reqs)
        r = self.stub.batchPredictText(batch_req)
        if r.status.success:
            return r.texts
        else:
            log.error("ocr rpc batchPredictText error: %s => %s",
                      r.status.error_code, r.status.error_msg)
            return None

    @staticmethod
    def get_element_name(e_type):
        return {
            ocr_pb2.Element.UNKNOWN: "unknown",
            ocr_pb2.Element.TEXT: "text",
            ocr_pb2.Element.LEGEND_TEXT: "legend_text",
            ocr_pb2.Element.LEGEND: "legend",
            ocr_pb2.Element.HAXIS: "haxis",
            ocr_pb2.Element.VAXIS: "vaxis",
            ocr_pb2.Element.COLUMN: "column",
            ocr_pb2.Element.BAR: "bar",
            ocr_pb2.Element.TITLE: "title",
            ocr_pb2.Element.FRONT_TEXTS: "front_texts"
        }.get(e_type, "unknown")


if __name__ == '__main__':
    import sys, codecs

    logging.basicConfig(level=logging.INFO)

    client = OcrClient(endpoint="127.0.0.1:12360")
    with codecs.open(sys.argv[1], "rb") as f:
        data1 = f.read()
    with codecs.open(sys.argv[1], "rb") as f:
        data2 = f.read()

    r = client.detectElement(data1)
    print(r)
    print(".....................\n")
    r = client.predictText(data2)
    print(r)
    print(".....................\n")
    r = client.batchPredictText([data1, data2])
    for x in r:
        print(x, end="\n*********\n")
    print()
