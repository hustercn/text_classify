# coding: utf-8
from __future__ import absolute_import
from __future__ import unicode_literals

from six import text_type
import io
import time
import traceback
import logging

from bson import json_util
from PIL import Image

from abcft_algorithm_grpc import algorithm_internal_rpc_pb2, algorithm_internal_rpc_pb2_grpc
from abcft_algorithm_grpc import ocr_pb2, ocr_pb2_grpc

from abcft_algorithm_grpc.reply import Reply
from abcft_algorithm_grpc.utils import get_file_data


log = logging.getLogger(__name__)


class BaseInternalHandler(algorithm_internal_rpc_pb2_grpc.AlgorithmInternalServiceServicer):
    def __init__(self):
        mm = [x for x in dir(self) if x.startswith("do_")]
        # m[3:] remove "do_"
        self._methods = {text_type(m[3:]): getattr(self, m) for m in mm}
        self._versions = []

    def Ping(self, request, context):
        reply = algorithm_internal_rpc_pb2.PingReply()
        reply.timestamp = int(time.time() * 1000)
        return reply

    def GetVersions(self, request, context):
        reply = algorithm_internal_rpc_pb2.VersionsReply()
        vv = []
        for v in self._versions:
            x = algorithm_internal_rpc_pb2.VersionInfo()
            x.name = v["name"]
            major, minor, patch = v["version"]
            x.version_major = major
            x.version_minor = minor
            x.version_patch = patch
            x.version = "{}.{}.{}".format(major, minor, patch)
            vv.append(x)
        reply.versions.extend(vv)
        return reply

    def ListMethods(self, request, context):
        reply = algorithm_internal_rpc_pb2.MethodsReply()
        mm = []
        for x in self._methods:
            m = algorithm_internal_rpc_pb2.MethodInfo()
            m.name = x
            m.desc = self._methods[x].__doc__ or ""
            mm.append(m)
        reply.methods.extend(mm)
        return reply


    def Request(self, request, context):
        method = request.method
        if request.url:
            filedata = get_file_data(request.url)
        else:
            filedata = request.data
        if request.params:
            params = json_util.loads(request.params)
        else:
            params = None
        version = request.version
        rr = self._do_request(method, filedata, params, version)
        for r in rr:
            yield r


    def _replay(self, r):
        replay = algorithm_internal_rpc_pb2.RpcReply()
        replay.ok = r.get("ok", False)
        replay.result = r.get("result", b"")
        replay.code = r.get("code", Reply.InternalError)
        replay.msg = r.get("msg", "")
        replay.extra = r.get("extra", "")
        return replay

    def _do_request(self, method, filedata, params, version):
        if method not in self._methods:
            yield self._replay(Reply.fail(Reply.NotImplemented))
            return

        do_method = self._methods[method]
        try:
            rr = do_method(filedata, params, version)
            for r in rr:
                yield self._replay(r)
        except Exception as e:
            err = traceback.format_exc()
            logging.error("%s", "".join(err))
            yield self._replay(Reply.fail(Reply.InternalError))


    def get_data_from_data_or_url(self, request):
        if request.data:
            return request.data
        else:
            return get_file_data(request.url)


class BaseOcrHandler(ocr_pb2_grpc.OcrServiceServicer):

    def get_image_from_request(self, request):
        if request.data:
            data = request.data
        else:
            data = get_file_data(request.url)

        image = Image.open(io.BytesIO(data))
        image = image.convert("RGB")
        return image
