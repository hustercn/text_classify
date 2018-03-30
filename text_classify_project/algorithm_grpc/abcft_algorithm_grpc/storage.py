# coding: utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import logging

from six import text_type, binary_type
import requests
import oss2
import qcloud_cos

from abcft_algorithm_grpc import utils

log = logging.getLogger(__name__)


class BaseStorage(object):
    def __init__(self, dev=False):
        self._dev = dev
        self._domains = {}

    def get_public_url(self, url):
        return url

    def get_internal_url(self, url):
        return url

    def ensure_url_access(self, url):
        return url

    def get_download_url(self, path):
        raise NotImplementedError("Method not implemented!")

    def upload(self, path, filedata):
        raise NotImplementedError("Method not implemented!")

    @utils.safe_run
    def download(self, url):
        if self._dev:
            url = self.get_public_url(url)
        else:
            url = self.get_internal_url(url)

        log.info("download %s", url)
        domain = url[url.find("//") + 2: url.find("/", 8)]
        if domain not in self._domains:
            self._domains[domain] = requests.session()
        r = self._domains[domain].get(url, timeout=30)
        if r.status_code != 200:
            log.warning("get status_code %d", r.status_code)
            return None
        return r.content


ALIYUN_OSS_REGION = "cn-hangzhou"
ALIYUN_OSS_BUCKET = "abc-crawler"
ALIYUN_OSS_ACCESS_KEY = "LTAITN0hCn7KBUzK"
ALIYUN_OSS_ACCESS_SECRET = "c8SOHjg15bkkW3AxQmbDyyDQA8fnNI"

DEFAULT_EXPIRES = 4 * 60 * 60

class _SimpleRequest(object):
    def __init__(self, url, method='GET'):
        self.url = url
        self.method = method
        self.headers = {}
        self.params = {}


class TemporaryAuth(oss2.Auth):
    def __init__(self, access_key_id, access_key_secret):
        super(TemporaryAuth, self).__init__(access_key_id, access_key_secret)

    def sign_url(self, region, bucket_name, oss_path, expires, internal_region, public_access):
        url = "http://{bucket_name}.oss-{region}{internal}.aliyuncs.com/{oss_path}".format(
            region=region,
            bucket_name=bucket_name,
            internal='-internal' if internal_region else '',
            oss_path=oss_path.strip('/')
        ).encode("utf-8")
        if not public_access:
            req = _SimpleRequest(url)
            return self._sign_url(req, bucket_name, oss_path, expires)
        else:
            return url


class AliyunStorage(BaseStorage):
    def __init__(self, **kwargs):
        dev = kwargs.pop("dev", False)
        super(AliyunStorage, self).__init__(dev=dev)

        self._region = kwargs.get("region", ALIYUN_OSS_REGION).encode("utf-8")
        self._bucket = kwargs.get("bucket", ALIYUN_OSS_BUCKET).encode("utf-8")
        if self._dev:
            self._upload_endpoint = "http://oss-{}.aliyuncs.com".format(self.region).encode("utf-8")
            self._download_endpoint = "http://{}.oss-{}.aliyuncs.com".format(self.bucket, self.region).encode("utf-8")
        else:
            self._upload_endpoint = "http://oss-{}-internal.aliyuncs.com".format(self.region).encode("utf-8")
            self._download_endpoint = "http://{}.oss-{}-internal.aliyuncs.com".format(self.bucket, self.region).encode("utf-8")

        access_key = kwargs.get("accessKey", ALIYUN_OSS_ACCESS_KEY)
        access_secret = kwargs.get("accessSecret", ALIYUN_OSS_ACCESS_SECRET)
        self._public_access = kwargs.get("publicAccess", True)
        self._auth = TemporaryAuth(access_key, access_secret)
        try:
            self._upload_bucket = oss2.Bucket(self._auth, self._upload_endpoint, self.bucket, connect_timeout=5)
            # just make the seesion (keep-alive)
            self._upload_bucket.head_object("__test__")
        except oss2.exceptions.NotFound:
            pass
        except Exception:
            log.warning("upload_bucket init error")

    @property
    def dev_mode(self):
        return self._dev

    @property
    def region(self):
        return self._region

    @property
    def bucket(self):
        return self._bucket

    @property
    def public_access(self):
        return self._public_access

    @property
    def upload_bucket(self):
        return self._upload_bucket

    def get_download_url(self, path_or_url):
        if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
            return path_or_url
        else:
            if path_or_url.startswith("/"):
                return self._download_endpoint + path_or_url
            else:
                return self._download_endpoint + "/" + path_or_url

    def get_public_url(self, url):
        return url.replace("-internal.aliyuncs.", ".aliyuncs.")

    def get_internal_url(self, url):
        if "internal" in url:
            return url
        else:
            return url.replace(".aliyuncs.", "-internal.aliyuncs.")

    def ensure_url_access(self, url):
        return self.sign_object_url(url)

    def sign_object_url(self, path_or_url, expires=DEFAULT_EXPIRES):
        if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
            url = path_or_url
            idx = url.find('/', url.find('//') + 2)
            oss_path = url[idx:]
        else:
            oss_path = path_or_url
        return self._auth.sign_url(
            self._region,
            self._bucket, oss_path,
            expires=expires,
            internal_region=not self._dev,
            public_access=self._public_access
        )

    @utils.safe_run
    def upload(self, path, filedata):
        if isinstance(filedata, text_type):
            self._upload_bucket.put_object_from_file(path, filedata)
        elif isinstance(filedata, binary_type):
            self._upload_bucket.put_object(path, filedata)
        else:
            raise TypeError("filedata should be filename(text_type) or content(binary_type)")



QCLOUD_COS_REGION = "__TODO__"
QCLOUD_COS_BUCKET = "__TODO__"
QCLOUD_COS_APP_ID = "__TODO__"
QCLOUD_COS_SECRET_ID = "__TODO__"
QCLOUD_COS_SECRET_KEY = "__TODO__"
QCLOUD_COS_EXPIRED = 3600 * 24 * 365 * 10  # 10 years


class QCloudStorage(BaseStorage):
    def __init__(self, **kwargs):
        dev = kwargs.pop("dev", False)
        super(QCloudStorage, self).__init__(dev=dev)

        self._region = kwargs.get("region", QCLOUD_COS_REGION)
        self._bucket = kwargs.get("bucket", QCLOUD_COS_BUCKET)
        self._public_access = kwargs.get("publicAccess", True)

        app_id = kwargs.get("app_id", QCLOUD_COS_APP_ID)
        secret_id = kwargs.get("secret_id", QCLOUD_COS_SECRET_ID)
        secret_key = kwargs.get("secret_key", QCLOUD_COS_SECRET_KEY)
        # FIXME: qcloud_cos api changes
        self._cos_client = qcloud_cos.CosClient(app_id, secret_id, secret_key, region=self.region)
        self._cos_config = qcloud_cos.CosConfig(sign_expired=QCLOUD_COS_EXPIRED, region=self.region)
        self._cos_client.set_config(self._cos_config)
        download_hostname = self._cos_config.get_download_hostname()
        self._download_endpoint = "http://{}-{}.{}".format(self.bucket, app_id, download_hostname)

    @property
    def dev_mode(self):
        return self._dev

    @property
    def region(self):
        return self._region

    @property
    def bucket(self):
        return self._bucket

    @property
    def public_access(self):
        return self._public_access

    def get_download_url(self, path_or_url):
        if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
            return path_or_url
        else:
            if path_or_url.startswith("/"):
                return self._download_endpoint + path_or_url
            else:
                return self._download_endpoint + "/" + path_or_url

    @utils.safe_run
    def upload(self, path, filedata):
        # qcloud_cos needs path to startswith /
        if not path.startswith("/"):
            path = "/" + path
        if isinstance(filedata, text_type):
            req = qcloud_cos.UploadFileRequest(self.bucket, path, filedata, insert_only=0)
            return self._cos_client.update_file(req)
        elif isinstance(filedata, binary_type):
            req = qcloud_cos.UploadFileFromBufferRequest(self.bucket, path, filedata, insert_only=0)
            return self._cos_client.upload_file_from_buffer(req)
        else:
            raise TypeError("filedata should be filename(text_type) or content(binary_type)")


_Storage = None  # type: BaseStorage

def init(storage, **kwargs):
    cls = {
        "aliyun_oss": AliyunStorage,
        "qcloud_cos": QCloudStorage,
    }.get(storage)
    if not cls:
        raise ValueError("no storage: {}".format(storage))

    global _Storage
    _Storage = cls(**kwargs)

def upload(path, filedata):
    if not _Storage:
        raise ValueError("storage not init yet")
    return _Storage.upload(path, filedata)

def download(path):
    if not _Storage:
        raise ValueError("storage not init yet")
    return _Storage.download(path)

def download_if_needed(filename):
    if isinstance(filename, text_type) and filename.startswith("http"):
        return download(filename)
    else:
        return filename

def ensure_url_access(url):
    if not _Storage:
        raise ValueError("storage not init yet")
    return _Storage.ensure_url_access(url)


if __name__ == '__main__':
    init("qcloud_cos",
         region="shanghai",
         bucket="parsedev01",
         app_id=1253835285,
         secret_id="AKIDAEdw3yZDzovr0EClohR1qaDgx39AolIe",
         secret_key="NYYLJAqKchbJsFQUM4ax87uVu6CAyd6N",
         )

    import sys
    import codecs

    filename = sys.argv[1]
    with codecs.open(filename, "rb") as f:
        filedata = f.read()

    r = upload("test", filename)
    print ("upload local file:", r)
    r = upload("/test", filedata)
    print("upload binary data:", r)

    print ("download")
    url = _Storage.get_download_url("/test")
    d = download(url)
    with codecs.open("test", "wb") as f:
        f.write(d)
