# coding: utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import os
import json
import codecs
import logging

import kazoo.client


SERVICE_PREFIX = "/abcft/algorithm_services"

log = logging.getLogger(__name__)


class ServiceAgent(object):
    def __init__(self, service_name, hosts='127.0.0.1:2181'):
        self.service_name = service_name
        self._distribute_path = os.path.join(SERVICE_PREFIX, "distribute")
        self._services_path = os.path.join(SERVICE_PREFIX, "services")
        self._running_path = os.path.join(SERVICE_PREFIX, "running")

        self._own_service_path = os.path.join(SERVICE_PREFIX, "services", self.service_name)
        self._own_running_path = os.path.join(SERVICE_PREFIX, "running", self.service_name)
        self._proxy_path = os.path.join(self._own_running_path, "proxy")

        self.zk = kazoo.client.KazooClient(hosts=hosts)
        self.zk.start()
        log.info("connected to zookeeper")


    def start_serve(self, endpoint):
        self.endpoint = endpoint
        self._endpoint_path = os.path.join(self._own_running_path, self.endpoint)
        self.zk.ensure_path(self._own_running_path)
        self.zk.create(self._endpoint_path, ephemeral=True)
        log.info("start_serve %s", self.service_name)

    def stop_serve(self):
        self.zk.delete(self._endpoint_path)
        log.info("stop_serve %s", self.service_name)

    def get_running_services(self):
        return [x for x in self.zk.get_children(self._own_running_path) if x != "proxy"]

    def get_configure(self):
        try:
            return self.get_service(self.service_name)
        except:
            log.error("no configure of service %s", self.service_name)
            return None


    def get_service(self, name):
        srv_path = os.path.join(self._services_path, name)
        if not self.zk.exists(srv_path):
            return None

        srv_data = self.zk.get(srv_path)[0]
        srv_data = codecs.decode(srv_data, "utf-8")
        srv = json.loads(srv_data)
        srv.update({"name": name})
        return srv

    def get_all_services(self):
        services = []
        services_chd = self.zk.get_children(self._services_path)
        for chd in services_chd:
            srv = self.get_service(chd)
            services.append(srv)
        return services


    def get_distribute(self, host):
        dst_path = os.path.join(self._distribute_path, host)
        if not self.zk.exists(dst_path):
            return None

        dst_data = self.zk.get(dst_path)[0]
        dst_data = codecs.decode(dst_data, "utf-8")
        dst = json.loads(dst_data)
        dst.update({"host": host})
        return dst

    def get_all_distribute(self):
        distribute = []
        distribute_chd = self.zk.get_children(self._distribute_path)
        for host in distribute_chd:
            dst = self.get_distribute(host)
            distribute.append(dst)
        return distribute

    def get_all_proxies(self):
        dists = self.get_all_distribute()
        proxies = {}
        for d in dists:
            for r in d["roles"]:
                if r.startswith("proxy_"):
                    name = r[6:]
                    proxies[name] = {
                        "name": name,
                        "host": d["ip"],
                    }
        srvs = self.get_all_services()
        for s in srvs:
            name = s["name"]
            if name in proxies:
                proxies[name]["port"] = int(s["base_port"])
        return [proxies[name] for name in proxies]


if __name__ == '__main__':
    sa = ServiceAgent("rpc_master", hosts="10.11.255.21:2181")
    proxies = sa.get_all_proxies()
    print (proxies)
