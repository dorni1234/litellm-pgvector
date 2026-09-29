import datetime, os
from fastapi import APIRouter, Depends, HTTPException, Form
from typing import Annotated, Optional
from sqlmodel import select, col, desc
from sqlalchemy import label
from uuid import UUID

from models import (
    VectorStoreResponse,
    VectorStoreCreateRequest,
    VectorStoreListResponse,
    VectorStoreSearchResponse,
    VectorStoreSearchRequest,
    SearchResult,
    ContentChunk,
    VectorStoreRetrieveResponse,
    VectorStoreDeleteResponse,
    VectorStoreUpdateRequest,
    VectorStoreUpdateResponse
)

from util import get_litellm_vkey_info

from embedding_service import embedding_service

from classes.database import VectorStore, database_instance, Embedding

router = APIRouter()

@router.post("/v1/vector_stores", response_model=VectorStoreResponse)
async def create_vector_store(
    request: VectorStoreCreateRequest, litellm_vkey_info = Depends(get_litellm_vkey_info),

):
    """
    Create a new vector store.
    """
    try:
        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']

        store = VectorStore(
            name=request.name,
            user_id=user,
            team_id=team,
            file_counts={"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            status="completed",
            usage_bytes=0,
            expires_after=request.expires_after,
            store_metadata=request.metadata or {}
        )

        session = database_instance.session()

        session.add(store)
        session.commit()

        # Convert to response format
        created_at = int(store.created_at.timestamp())
        expires_at = (
            int(store.expires_at.timestamp())
            if store.expires_at
            else None
        )
        last_active_at = (
            int(store.last_active_at.timestamp())
            if store.last_active_at
            else None
        )

        session.close()

        return VectorStoreResponse(
            id=store.id.hex,
            created_at=created_at,
            name=store.name,
            usage_bytes=store.usage_bytes or 0,
            file_counts=store.file_counts
            or {
                "in_progress": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 0,
            },
            status=store.status,
            expires_after=store.expires_after,
            expires_at=expires_at,
            last_active_at=last_active_at,
            metadata=store.store_metadata
        )

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500, detail=f"Failed to create vector store: {str(e)}"
        )

@router.get("/v1/vector_stores", response_model=VectorStoreListResponse)
async def list_vector_stores(
    limit: Optional[int] = 20,
    after: Optional[str] = None,
    before: Optional[str] = None,
    litellm_vkey_info = Depends(get_litellm_vkey_info),
):
    """
    List vector stores with optional pagination.
    """
    try:
        limit = min(limit or 20, 100)  # Cap at 100 results

        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']

        statement = select(VectorStore)

        if after:
            statement = statement.where(col(VectorStore.id) > UUID(after))

        if before:
            statement = statement.where(col(VectorStore.id) < UUID(before))

        if team:
            statement = statement.where(VectorStore.team_id == team)
        elif user:
            statement = statement.where(VectorStore.user_id == user)

        statement = statement.order_by(desc(VectorStore.created_at)).limit(limit + 1)

        session = database_instance.session()
        res = session.exec(statement)
        stores = res.all()
        session.close()

        # Check if there are more results
        has_more = len(stores) > limit
        if has_more:
            stores = stores[:limit]  # Remove extra result

        # Convert to response format
        vector_stores = []
        for store in stores:
            created_at = int(store.created_at.timestamp())
            expires_at = int(store.expires_at.timestamp()) if store.expires_at else None
            last_active_at = int(store.last_active_at.timestamp()) if store.last_active_at else None

            vector_store = VectorStoreResponse(
                id=store.id.hex,
                created_at=created_at,
                name=store.name,
                usage_bytes=store.usage_bytes or 0,
                file_counts=store.file_counts
                or {
                    "in_progress": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "total": 0,
                },
                status=store.status,
                expires_after=store.expires_after,
                expires_at=expires_at,
                last_active_at=last_active_at,
                metadata=store.store_metadata,
            )
            vector_stores.append(vector_store)

        first_id = vector_stores[0].id if vector_stores else None
        last_id = vector_stores[-1].id if vector_stores else None

        return VectorStoreListResponse(
            data=vector_stores, first_id=first_id, last_id=last_id, has_more=has_more
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to list vector stores: {str(e)}"
        )

@router.post(
    "/v1/vector_stores/{vector_store_id}/search",
    response_model=VectorStoreSearchResponse,
)
@router.post(
    "/vector_stores/{vector_store_id}/search", response_model=VectorStoreSearchResponse
)
async def search_vector_store(
    vector_store_id: str,
    request: VectorStoreSearchRequest,
    litellm_vkey_info = Depends(get_litellm_vkey_info),
    ):
    """
    Search a vector store for similar content.
    """
    try:
        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']

        statement = select(VectorStore).where(col(VectorStore.id) == UUID(vector_store_id))
        if team:
            statement = statement.where(VectorStore.team_id == team)
        elif user:
            statement = statement.where(VectorStore.user_id == user)
        else:
            raise HTTPException(status_code=401, detail="No valid credentials provided")

        session = database_instance.session()
        res = session.exec(statement)
        store = res.first()
        if not store:
            raise HTTPException(status_code=404, detail="Vector store not found")

        # Generate embedding for query
        query_embedding = await embedding_service.generate_embedding(request.query, litellm_vkey_info['key'])
        query_embedding = query_embedding + [0] * (3072 - len(query_embedding))

        # Build the raw SQL query for vector similarity search
        limit = min(request.limit or 20, 100)  # Cap at 100 results

        embedding_statement = select(
            Embedding.id, 
            Embedding.content, 
            Embedding.embedding_metadata, 
            Embedding.embedding.l2_distance(query_embedding).label('distance') # pyright: ignore[reportAttributeAccessIssue]
            ).where(col(VectorStore.id) == UUID(vector_store_id))

        if request.filters:
            for key, value in request.filters.items():
                embedding_statement = embedding_statement.where(col(Embedding.embedding_metadata)[key].as_string() == value)


        embedding_statement = embedding_statement.order_by(label('distance', col(Embedding.embedding)).asc()).limit(limit)
        embedding_results = session.exec(embedding_statement).all()

        # Convert results to SearchResult objects
        search_results = []
        for embedding_result in embedding_results:
            # Convert distance to similarity score (1 - normalized_distance)
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            similarity_score = max(0, 1 - (embedding_result[3] / 2))

            # Extract filename from metadata or use a default
            metadata = embedding_result[2] or {}
            if not metadata:
                filename = "document.txt"
            else:
                filename = metadata.get("filename", "document.txt") # pyright: ignore[reportAttributeAccessIssue]

            content_chunks = [ContentChunk(type="text", text=embedding_result[1])]

            result = SearchResult(
                file_id=embedding_result[0].hex,
                filename=filename,
                score=similarity_score,
                attributes=metadata if (request.return_metadata and metadata) else None, # pyright: ignore[reportArgumentType]
                content=content_chunks,
            )
            search_results.append(result)

        return VectorStoreSearchResponse(
            search_query=request.query,
            data=search_results,
            has_more=False,  # TODO: Implement pagination
            next_page=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.delete("/vector_stores/{vector_store_id}/", response_model=VectorStoreDeleteResponse)
async def delete_vector_store(vector_store_id: str, litellm_vkey_info = Depends(get_litellm_vkey_info)):
    try:
        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']

        statement = select(VectorStore).where(col(VectorStore.id) == UUID(vector_store_id))
        if team:
            statement = statement.where(VectorStore.team_id == team)
        elif user:
            statement = statement.where(VectorStore.user_id == user)
        else:
            raise HTTPException(status_code=401, detail="No valid credentials provided")

        session = database_instance.session()
        res = session.exec(statement)
        store = res.first()

        if not store:
            raise HTTPException(status_code=404, detail="Vector store not found")

        session.delete(store)
        session.commit()
        session.close()

        return VectorStoreDeleteResponse(id=store.id.hex)

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/vector_stores/{vector_store_id}/", response_model=VectorStoreRetrieveResponse)
async def retrieve_vector_store(vector_store_id: str, litellm_vkey_info = Depends(get_litellm_vkey_info)):
    try:
        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']

        statement = select(VectorStore).where(col(VectorStore.id) == UUID(vector_store_id))
        if team:
            statement = statement.where(VectorStore.team_id == team)
        elif user:
            statement = statement.where(VectorStore.user_id == user)
        else:
            raise HTTPException(status_code=401, detail="No valid credentials provided")

        session = database_instance.session()
        res = session.exec(statement)
        store = res.first()
        session.close()
        if not store:
            raise HTTPException(status_code=404, detail="Vector store not found")

        return VectorStoreRetrieveResponse(
            id=store.id.hex,
            name=store.name,
            created_at=int(store.created_at.timestamp()),
            file_counts=store.file_counts,
            usage_bytes=store.usage_bytes or 0
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/vector_stores/{vector_store_id}/", response_model=VectorStoreUpdateRequest)
async def update_vector_store(vector_store_id: str, request: VectorStoreUpdateRequest, litellm_vkey_info = Depends(get_litellm_vkey_info)):
    try:
        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']

        statement = select(VectorStore).where(col(VectorStore.id) == UUID(vector_store_id))
        if team:
            statement = statement.where(VectorStore.team_id == team)
        elif user:
            statement = statement.where(VectorStore.user_id == user)
        else:
            raise HTTPException(status_code=401, detail="No valid credentials provided")

        session = database_instance.session()
        res = session.exec(statement)
        store = res.first()
        if not store:
            raise HTTPException(status_code=404, detail="Vector store not found")

        if request.name:
            store.name = request.name

        if request.expires_after:
            store.expires_after = request.expires_after

        if request.metadata:
            store.store_metadata = request.metadata

        session.add(store)
        session.commit()
        session.close()

        return VectorStoreRetrieveResponse(
            id=store.id.hex,
            name=store.name,
            created_at=int(store.created_at.timestamp()),
            file_counts=store.file_counts,
            usage_bytes=store.usage_bytes or 0
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")