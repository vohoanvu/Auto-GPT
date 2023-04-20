""" Zilliz memory storage provider."""
from pymilvus import connections

from autogpt.memory.milvus import MilvusMemory


class ZillizMemory(MilvusMemory):
    """Zilliz memory storage provider, wrap the milvus memory.
    Wraps the MilvusMemory to show correct memory type.
    """

    def __init__(self, cfg) -> None:
        # use cloud service by remote uri.
        connections.connect(
            uri=cfg.zilliz_uri,
            user=cfg.milvus_username,
            password=cfg.milvus_password,
            secure=True,
        )
        self.collection_name = cfg.milvus_collection
        self.index_params = {
            "index_type": "AUTOINDEX",
            "metric_type": "IP",
            "params": {},
        }

        # init collection.
        self.init_collection()