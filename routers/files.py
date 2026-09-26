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
from classes.database import database_instance, File
from sqlmodel import select, col
from uuid import UUID

router = APIRouter()

@router.post(
    "/files",
    response_model=UploadFileResponse,
)
async def upload_file(
    data: Annotated[UploadFileRequest, Form()],
    litellm_vkey_info = Depends(get_litellm_vkey_info),
):
    try:
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

        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']
        key_user_or_team_type = "user" if user else "team"
        key_user_or_team_id = user if user else team
        upload_res = await s3_file_handler.upload_file(file, f"{key_user_or_team_type}/{key_user_or_team_id}")

        session = database_instance.session()
        new_file = File(
            user_id=user,
            team_id=team,
            filename=file.filename,
            size=file.size,
            purpose=data.purpose,
            created_at=datetime.datetime.now(),
            expires_at=None,
            filename_on_disk=upload_res['s3_key']
        )

        session.add(new_file)
        session.commit()
        session.refresh(new_file)
        session.close()

        return UploadFileResponse(
            id=new_file.id.hex,
            bytes=file.size,
            created_at=int(new_file.created_at.timestamp()),
            filename=file.filename,
            purpose=data.purpose,
            object="file",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to create embedding: {str(e)}"
        )

@router.delete("/v1/files/{file_id}", response_model=DeleteFileResponse)
async def delete_file(
    file_id: str,
    litellm_vkey_info = Depends(get_litellm_vkey_info)    
):
    """
    Delete a file, from storage aswell as from all vector stores.
    """
    try:
        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']

        session = database_instance.session()
        statement = select(File).where(col(File.id) == UUID(file_id))

        if user:
            statement = statement.where(File.user_id == user)
        else:
            statement = statement.where(File.team_id == team)

        file = session.exec(statement).first()
        if not file:
            raise HTTPException(status_code=404, detail="File not found")

        # TODO replace with call that creates a s3 or local file handler, depending on configuration
        s3_file_handler = S3FileHandler(
            region=settings.s3_region,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            bucket=settings.s3_bucket,
            host=settings.s3_host
            )

        await s3_file_handler.delete_file(file.filename_on_disk)

        session.delete(file)
        session.commit()
        session.close()

        # TODO update vector store statistics

        return DeleteFileResponse(id=file_id)

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to create embedding: {str(e)}"
        )