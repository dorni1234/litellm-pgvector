from abc import ABC, abstractmethod, abstractproperty
from fastapi import UploadFile


class AbstractFileHandler(ABC):
    @classmethod
    @abstractmethod
    def upload_file(cls, file: UploadFile, subdirectory: str):
        pass

    @classmethod
    @abstractmethod
    def get_file(cls, file: str):
        pass
