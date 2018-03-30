# coding: utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

from six import text_type
import codecs
import logging

from bson import json_util
import grpc
import google.protobuf.empty_pb2 as empty_pb2

from abcft_algorithm_grpc import algorithm_internal_rpc_pb2, algorithm_internal_rpc_pb2_grpc

from abcft_algorithm_grpc.utils import load_certificate
from abcft_algorithm_grpc.server import DEFAULT_HOST, DEFAULT_PORT
from abcft_algorithm_grpc.server import DEFAULT_CERT_PATH, DEFAULT_CERT_CA_NAME
from abcft_algorithm_grpc.server import MAX_RECEIV_MESSAGE_SIZE


DEFAULT_CLIENT_CERT_NAME = "grpc-client"

log = logging.getLogger(__name__)


class BaseClient(object):
    #TODO: remove host and port
    def __init__(self, stub, service_name=None, host=DEFAULT_HOST, port=DEFAULT_PORT, endpoint=None,
                 cert_path=DEFAULT_CERT_PATH, cert_name=DEFAULT_CLIENT_CERT_NAME, cert_ca_name=DEFAULT_CERT_CA_NAME):
        if endpoint:
            self.endpoint = endpoint
        else:
            self.endpoint = "{}:{}".format(host, port)

        private_key_cert = load_certificate(cert_path, "{}.pem".format(cert_name))
        cert_chain = load_certificate(cert_path, "{}.crt".format(cert_name))
        root_cert = load_certificate(cert_path, "{}.pem".format(cert_ca_name))

        credentials = grpc.ssl_channel_credentials(
            root_certificates=root_cert,
            private_key=private_key_cert,
            certificate_chain=cert_chain
        )
        if not service_name:
            service_name = "grpc"
        target_name = "{}.rpc.modeling.ai".format(service_name)
        self.channel = grpc.secure_channel(
            self.endpoint, credentials,
            options=(
                ('grpc.ssl_target_name_override', target_name),
                ('grpc.max_receive_message_length', MAX_RECEIV_MESSAGE_SIZE),
            )
        )
        self.stub = stub(self.channel)
        log.info("rpc connected to %s", self.endpoint)


class BaseInternalClient(BaseClient):
    def __init__(self, service_name=None, host=DEFAULT_HOST, port=DEFAULT_PORT, endpoint=None,
                 cert_path=DEFAULT_CERT_PATH, cert_name=DEFAULT_CLIENT_CERT_NAME, cert_ca_name=DEFAULT_CERT_CA_NAME):
        super(BaseInternalClient, self).__init__(
            algorithm_internal_rpc_pb2_grpc.AlgorithmInternalServiceStub,
            service_name=service_name, host=host, port=port, endpoint=endpoint,
            cert_path=cert_path, cert_name=cert_name, cert_ca_name=cert_ca_name
        )

    def _reply(self, reply):
        return {
            "ok": reply.ok,
            "result": json_util.loads(codecs.decode(reply.result, "utf-8")),
            "code": reply.code,
            "msg": reply.msg,
            "extra": reply.extra
        }

    def Ping(self):
        return self.stub.Ping(empty_pb2.Empty())

    def GetVersions(self):
        return self.stub.GetVersions(empty_pb2.Empty())

    def ListMethods(self):
        return self.stub.ListMethods(empty_pb2.Empty())

    def Request(self, method, data_or_url, version=None, **kwargs):
        req = algorithm_internal_rpc_pb2.RpcRequest()
        req.method = method
        if isinstance(data_or_url, text_type):
            req.url = data_or_url
        else:
            req.data = data_or_url
        if kwargs:
            req.params = json_util.dumps(kwargs, ensure_ascii=False).encode("utf-8")
        if version:
            req.version = version
        try:
            rr = self.stub.Request(req)
            for r in rr:
                yield self._reply(r)
        except Exception as e:
            log.exception(e)
            yield None

    def ImageDetect(self, url_or_data):
        req = algorithm_internal_rpc_pb2.ImageDetectRequest()
        if isinstance(url_or_data, text_type):
            req.url = url_or_data
        else:
            req.data = url_or_data
        try:
            rr = self.stub.ImageDetect(req)
            for r in rr:
                yield {
                    "ok": r.ok,
                    "code": r.code,
                    "msg": r.msg,
                    "type": r.type,
                    "score": r.score,
                    "bbox": [float(x) for x in r.bbox],
                }
        except Exception as e:
            log.exception(e)
            yield None

    def ImageClassify(self, url_or_data):
        req = algorithm_internal_rpc_pb2.ImageClassifyRequest()
        if isinstance(url_or_data, text_type):
            req.url = url_or_data
        else:
            req.data = url_or_data
        try:
            rr = self.stub.ImageClassify(req)
            for r in rr:
                yield {
                    "ok": r.ok,
                    "code": r.code,
                    "msg": r.msg,
                    "type": r.type,
                    "score": r.score,
                }
        except Exception as e:
            log.exception(e)
            yield None

    def ImageClassifyBatch(self, urls):
        request = []
        for url in urls:
            url_or_data = url.get("file", "")
            _id = url.get("_id", "")
            req = algorithm_internal_rpc_pb2.ImageClassifyRequest(id=_id)
            if isinstance(url_or_data, text_type):
                req.url = url_or_data
            else:
                req.data = url_or_data
            request.append(req)
        try:
            rr = self.stub.ImageClassifyBatch(iter(request))
            for r in rr:
                yield {
                    "ok": r.ok,
                    "code": r.code,
                    "msg": r.msg,
                    "result": r.result,
                    "id": r.id
                }
        except Exception as e:
            log.exception(e)
            yield None

    def TitleClassify(self, category, first_class, title, text):
        req = algorithm_internal_rpc_pb2.TitleClassifyRequest(
            category=category,
            first_class=first_class,
            text=text,
            title=title
        )
        try:
            rr = self.stub.TitleClassify(req)
            for r in rr:
                yield {
                    "ok": r.ok,
                    "code": r.code,
                    "msg": r.msg,
                    "classify_code": r.classify_code,
                    "classify_class1": r.classify_class1,
                    "classify_class2": r.classify_class2,
                    "classify_class3": r.classify_class3,
                }
        except Exception as e:
            log.exception(e)
            yield None

    def BitmapTableParse(self, url_or_data, table_type, use_ocr=True, format="json"):
        req = algorithm_internal_rpc_pb2.BitmapTableRequest(
            table_type=table_type,
            use_ocr=use_ocr,
            format=format,
        )
        if isinstance(url_or_data, text_type):
            req.url = url_or_data
        else:
            req.data = url_or_data
        try:
            rr = self.stub.BitmapTableParse(req)
            for r in rr:
                yield {
                    "ok": r.ok,
                    "code": r.code,
                    "msg": r.msg,
                    "parsed": r.parsed,
                }
        except Exception as e:
            log.exception(e)
            yield None

    def BitmapTableParseBatch(self, url_or_data, use_ocr=True, format="json",
                              ocr_engine=None, options=7):
        req = algorithm_internal_rpc_pb2.BitmapTableRequest(
            use_ocr=use_ocr,
            format=format,
            ocr_engine=ocr_engine,
            options=options
        )
        if isinstance(url_or_data, text_type):
            req.url = url_or_data
        else:
            req.data = url_or_data
        try:
            rr = self.stub.BitmapTableParseBatch(req)
            for r in rr:
                yield {
                    "ok": r.ok,
                    "code": r.code,
                    "msg": r.msg,
                    "parsed": r.parsed,
                    "bbox": [float(x) for x in r.bbox],
                }
        except Exception as e:
            log.exception(e)
            yield None


    def BitmapChartParse(self, url_or_data, chart_types):
        req = algorithm_internal_rpc_pb2.BitmapChartRequest(
            chart_types=chart_types
        )
        if isinstance(url_or_data, text_type):
            req.url = url_or_data
        else:
            req.data = url_or_data
        try:
            rr = self.stub.BitmapChartParse(req)
            for r in rr:
                yield {
                    "ok": r.ok,
                    "code": r.code,
                    "msg": r.msg,
                    "parsed": r.parsed,
                }
        except Exception as e:
            log.exception(e)
            yield None

    def TesseractDetectText(self, url_or_data):
        req = algorithm_internal_rpc_pb2.TesseractRequest()
        if isinstance(url_or_data, text_type):
            req.url = url_or_data
        else:
            req.data = url_or_data
        try:
            rr = self.stub.TesseractDetectText(req)
            for r in rr:
                yield {
                    "ok": r.ok,
                    "code": r.code,
                    "msg": r.msg,
                    "text": r.text,
                    "type": r.type,
                    "bbox": [float(x) for x in r.bbox]
                }
        except Exception as e:
            log.exception(e)
            yield None

    def ChartDetect(self, url_or_data):
        log.info("ImageChartDetect")
        req = algorithm_internal_rpc_pb2.ChartDetectRequest()
        if isinstance(url_or_data, text_type):
            req.url = url_or_data
        else:
            req.data = url_or_data
        try:
            rr = self.stub.ChartDetect(req)
            for r in rr:
                print (r)
                yield {
                    "ok": r.ok,
                    "code": r.code,
                    "msg": r.msg,
                    "score": r.score,
                    "type": r.type,
                    "bbox": [float(x) for x in r.bbox]
                }
        except Exception as e:
            log.exception(e)
            yield None


    def TextClassify(self, fulltext):
        req = algorithm_internal_rpc_pb2.TextClassifyRequest(
            fulltext=fulltext
        )
        try:
            rr = self.stub.TextClassify(req)
            for r in rr:
                yield {
                    "ok": r.ok,
                    "code": r.code,
                    "msg": r.msg,
                    "language": r.language,
                    "type": r.type,
                }
        except Exception as e:
            log.exception(e)
            yield None

    def Allennlp(self, passage,question):
        req = algorithm_internal_rpc_pb2.AllennlpRequest(
            passage=passage,
            question=question
        )
        try:
            rr = self.stub.Allennlp(req)
            for r in rr:
                yield {
                    "ok": r.ok,
                    "code": r.code,
                    "msg": r.msg,
                    "result": r.result,
                }
        except Exception as e:
            log.exception(e)
            yield None

if __name__ == '__main__':
    import sys
    import codecs

    logging.basicConfig(level=logging.INFO)
    client = BaseInternalClient(
        endpoint="localhost:12390",
        service_name="grpc",
        cert_path="/opt/algorithm/etc/grpc_certs",
        cert_name="grpc-client"
    )
    # r = client.Ping()
    # print (r)
    # r = client.GetVersions()
    # print(r)
    # r = client.ListMethods()
    # print(r)

    # with codecs.open(sys.argv[1], "rb") as f:
    #     data = f.read()
    #
    # rr = client.BitmapTableParse(data, 0, format="xml")
    # for r in rr:
    #     print (r["parsed"])
    #     print ()

    # urlll = u"http://img4.imgtn.bdimg.com/it/u=3568705674,2513558427&fm=214&gp=0.jpg"
    # ru = client.ImageClassify(urlll)
    # for r in ru:
    #     print(r)

    # data_dir = '/home/zhwpeng/nlp/data/texts/2257774.txt'
    # data_dir = '/home/zhwpeng/abc/nlp/data/little_data/notice/3861.txt'
    from glob import glob

    base_dir1 = '/home/zhwpeng/abc/nlp/data/raw_data/notice'
    base_dir2 = '/home/zhwpeng/abc/nlp/data/raw_data/report'
    notice_dirs = glob(base_dir1+'/*')
    data_dir = '/home/zhwpeng/abc/classify/gonggao/txt/58a4762f581e3071f0ff254a.txt'

    # with codecs.open(sys.argv[1], "r") as f:
    with codecs.open(data_dir, "r") as f:
        data = f.read()

    rr = client.TextClassify(data)
    for r in rr:
        print (r)
        print (r['type'])
