# coding: utf-8
from __future__ import unicode_literals

from bson import json_util


class Reply(object):
    InternalError = 1000
    NotImplemented = 1001
    ParamsRequired = 1002
    GetImageError = 1003
    UnSupportedType = 1004
    GetPdfError = 1005
    JavaSideError = 1006
    NoResult = 1007
    ParseChartError = 1008
    ParseTableError = 1009
    NotFinished = 1010
    JavaServerUnavailable = 1011

    @staticmethod
    def msg(code):
        return {
            Reply.InternalError: "InternalError",
            Reply.NotImplemented: "NotImplemented",
            Reply.ParamsRequired: "ParamsRequired",
            Reply.GetImageError: "GetImageError",
            Reply.UnSupportedType: "UnsupportedType",
            Reply.GetPdfError: "GetPdfError",
            Reply.JavaSideError: "JavaSideError",
            Reply.NoResult: "NoResult",
            Reply.ParseChartError: "ParseChartError",
            Reply.ParseTableError: "ParseTableError",
            Reply.NotFinished: "NotFinished",
            Reply.JavaServerUnavailable: "Java GRPC server unavailable"
        }.get(code, "Unknown")

    @staticmethod
    def _reply(ok, result=None, code=None, msg=None, extra=None):
        return {
            "ok": ok,
            "result": json_util.dumps(result, ensure_ascii=False).encode("utf-8"),
            "extra": json_util.dumps(extra, ensure_ascii=False) or "",
            "code": code,
            "msg": msg or ""
        }

    @staticmethod
    def ok(r, code=None, extra=None):
        return Reply._reply(
            True,
            result=r,
            code=code or 0,
            msg=Reply.msg(code) if code else None,
            extra=extra
        )

    @staticmethod
    def fail(c, msg=None):
        return Reply._reply(
            False,
            code=c,
            msg=msg or Reply.msg(c)
        )
