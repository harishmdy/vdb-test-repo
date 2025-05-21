from pathlib import Path
from typing import TypedDict


class BucketConfig(TypedDict):
    path: Path
    bucket_alias: str


EXAMPLE_COS_BUCKET: BucketConfig = {
    'path': Path('.') / 'config' / 'cos.yaml',
    'bucket_alias': 'example-bucket',
}
