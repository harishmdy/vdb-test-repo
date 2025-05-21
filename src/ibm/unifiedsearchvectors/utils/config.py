"""
Generic data classes to be shared across ETL pipeline classes.
These will rarely be edited
"""

from typing import TypedDict

from pydantic import BaseModel


class ObjectStorageConfig(BaseModel):
    """Configuration for object storage access."""

    access_key: str
    secret_key: str
    bucket: str
    endpoint_url: str


class ObjectStorageDictConfig(TypedDict):
    """Configuration for object storage access."""

    access_key: str
    secret_key: str
    bucket: str
    endpoint_url: str
    