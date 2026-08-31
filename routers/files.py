import datetime
from fastapi import APIRouter, Depends, HTTPException, Form
from typing import Annotated

from models import (
    UploadFileRequest,
    UploadFileResponse,
    DeleteFileResponse
)

from classes.s3_file_handler import S3FileHandler
from util import get_litellm_vkey_info
from config import settings
from classes.database import database_instance

router = APIRouter()

@router.post(
    "/files",
    response_model=UploadFileResponse,
)
async def upload_file(
    data: Annotated[UploadFileRequest, Form()],
    litellm_vkey_info = Depends(get_litellm_vkey_info),
):
    file = data.file
    if not file.size or not file.filename:
        raise HTTPException(status_code=500, detail="File upload failed")

    # TODO replace with call that creates a s3 or local file handler, depending on configuration
    s3_file_handler = S3FileHandler(
        region=settings.s3_region,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        bucket=settings.s3_bucket,
        host=settings.s3_host
        )

    key_user_or_team_type = "team" if litellm_vkey_info['info']['team_id'] else "user"
    key_user_or_team_id = litellm_vkey_info['info']['team_id'] if litellm_vkey_info['info']['team_id'] else litellm_vkey_info['info']['user_id']
    upload_res = await s3_file_handler.upload_file(file, f"{key_user_or_team_type}/{key_user_or_team_id}")

    print(upload_res)

    dbobject = await database_instance._db.file.create(
        data={
            "user_id": litellm_vkey_info['info']['user_id'] if not litellm_vkey_info['info']['team_id'] else None,
            "team_id": litellm_vkey_info['info']['team_id'],
            "filename": file.filename,
            "size": file.size,
            "purpose": data.purpose,
            "created_at": datetime.datetime.now(),
            "expires_at": None,
            "filename_on_disk": upload_res['s3_key'],
        }
    )

    return UploadFileResponse(
        id=dbobject.id,
        bytes=file.size,
        created_at=dbobject.created_at.strftime("%s"),
        filename=file.filename,
        purpose=data.purpose,
        object="file",
    )

@router.delete("/v1/files/{file_id}", response_model=DeleteFileResponse)
async def delete_file(
    file_id: str,
    litellm_vkey_info = Depends(get_litellm_vkey_info)    
):
    """
    Delete a file, from storage aswell as from all vector stores.
    """
    if litellm_vkey_info['info']['team_id']:
        get_file_query = f"""SELECT id, filename_on_disk FROM "File" WHERE id = $1 AND team_id = $2"""
        get_file_result = await database_instance._db.query_raw(get_file_query, file_id, litellm_vkey_info['info']['team_id'])
    elif litellm_vkey_info['info']['user_id']:
        get_file_query = f"""SELECT id, filename_on_disk FROM "File" WHERE id = $1 AND user_id = $2"""
        get_file_result = await database_instance._db.query_raw(get_file_query, file_id, litellm_vkey_info['info']['user_id'])
    else:
        raise HTTPException(status_code=401)

    print(get_file_result)
    if not get_file_result:
        raise HTTPException(status_code=404, detail="File not found")

    # TODO replace with call that creates a s3 or local file handler, depending on configuration
    s3_file_handler = S3FileHandler(
        region=settings.s3_region,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        bucket=settings.s3_bucket,
        host=settings.s3_host
        )

    await s3_file_handler.delete_file(get_file_result[0]['filename_on_disk'])

    if litellm_vkey_info['info']['team_id']:
        base_query = f"""DELETE FROM "File" WHERE id = $1 AND team_id = $2"""
        deletion_query_result = await database_instance._db.query_raw(base_query, file_id, litellm_vkey_info['info']['team_id'])
    elif litellm_vkey_info['info']['user_id']:
        base_query = f"""DELETE FROM "File" WHERE id = $1 AND user_id = $2"""
        deletion_query_result = await database_instance._db.query_raw(base_query, file_id, litellm_vkey_info['info']['user_id'])

    return DeleteFileResponse(id=file_id)