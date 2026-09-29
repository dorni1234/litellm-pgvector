import os
import uuid
from .abstract_file_handler import AbstractFileHandler
from fastapi import UploadFile, HTTPException
from pathlib import Path
from hashlib import sha256
from light_s3_client import Client
from typing import Optional
from io import BytesIO

class S3FileHandler(AbstractFileHandler):
    _s3_client: Client
    _s3_bucket: str

    @classmethod
    def __init__(self, region: str, access_key: str, secret_key: str, bucket: str, host: Optional[str] = None):
        self._s3_client = Client(
            region="us-west-1",
            access_key=access_key,
            secret_key=secret_key,
            server=host
            )

        self._s3_bucket = "vectorstore"


    @classmethod
    async def upload_file(self, file: UploadFile, subdirectory: str):
        key = f"{uuid.uuid4().hex}_{file.filename}"
        key_with_subdir = os.path.join(subdirectory, key)



        try:
            file.file.seek(0)
            contents = await file.read()
            res = self._s3_client.upload_fileobj(
                Fileobj=BytesIO(contents),
                Bucket="vectorstore",
                Key=key_with_subdir
                )
            print(res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")
        finally:
            await file.close()

        return {
            "filename": file.filename,
            "s3_key": key_with_subdir,
            "bucket": self._s3_bucket
        }

    @classmethod
    async def get_file(self, file: str) -> str:
        exists = self._s3_client.get_object(self._s3_bucket, file)
        if not exists:
            raise HTTPException(status_code=404, detail="File not found")

        Path("/tmp/litellm-vectordb-connector/files/").mkdir(parents=True, exist_ok=True)
        tmp_filename_on_disk = sha256(file.encode('utf-8')).hexdigest()
        filename, file_extension = os.path.splitext(file)
        tmp_filepath_on_disk = os.path.join("/tmp/litellm-vectordb-connector/files/", tmp_filename_on_disk) + file_extension
        return self._s3_client.download_file(self._s3_bucket, file, tmp_filepath_on_disk)

    @classmethod
    async def delete_file(self, file: str) -> bool:
        exists = self._s3_client.get_object(self._s3_bucket, file)
        if not exists:
            raise HTTPException(status_code=404, detail="S3 File not found")

        return self._s3_client.delete_file(self._s3_bucket, file)