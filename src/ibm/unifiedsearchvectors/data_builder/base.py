import pandas as pd
from carrot.data_builder import DataBuilderBase


class MitaDataBuilder(DataBuilderBase):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.data: pd.DataFrame

    def is_transformed(self) -> bool:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        raise NotImplementedError

    def clean(self):
        raise NotImplementedError

    def describe(self):
        raise NotImplementedError

    def split(self, *args, **kwargs):
        raise NotImplementedError

    def pre_transform_validate(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

    def post_transform_validate(self):
        raise NotImplementedError

    def run(self):
        ...

    def save(self, dataname: str, dir: str):
        raise NotImplementedError
