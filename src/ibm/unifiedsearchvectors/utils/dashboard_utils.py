import datetime

import pandas as pd
from pymilvus import (
    MilvusClient,
)


def convert_milvus_type(in_type):
    if in_type == 22:
        return "Array"
    elif in_type == 21:
        return "Varchar"
    elif in_type == 1:
        return "Boolean"
    elif in_type == 3:
        return "Int16"
    elif in_type == 5:
        return "Int64"
    elif in_type == 101:
        return "FloatVector"


class DashboardUtil:
    def __init__(self, milvus_client: MilvusClient):
        super().__init__()
        self.client = milvus_client

    def get_collection_summaries(self) -> pd.DataFrame:
        format = '%Y_%m_%d'

        # collection_obj.get_replicas()

        collections = self.client.list_collections()
        metadata = []
        for collection_name in collections:
            res = self.client.describe_collection(collection_name=collection_name)
            res['status'] = (
                self.client.get_load_state(collection_name=collection_name)
                .get('state')
                .name
            )
            stats = self.client.get_collection_stats(collection_name)
            res['approx_count'] = stats.get('row_count', -1)
            sp = str(res['description']).split('LAST_UPDATE_DATE: ')
            if len(sp) > 1:
                last_update_Str = str(res['description']).split('LAST_UPDATE_DATE: ')[1]
            else:
                last_update_Str = "1999_01_01"
            last_update = datetime.datetime.strptime(last_update_Str, format)
            res['last_update'] = last_update.date()
            metadata.append(res)
        df = pd.DataFrame(metadata)
        return df
