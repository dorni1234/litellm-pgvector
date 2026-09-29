import os
import time
import uuid
import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import Annotated


from models import (
    VectorStoreCreateRequest,
    VectorStoreResponse,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
    SearchResult,
    VectorStoreListResponse,
    ContentChunk,
)

from config import settings
from embedding_service import embedding_service
from util import get_litellm_vkey_info
from routers import files
from classes.database import database_instance, VectorStore, Embedding
from sqlmodel import select, col, text, desc
from sqlalchemy import label

load_dotenv()

app = FastAPI(
    title="OpenAI Vector Stores API",
    description="OpenAI-compatible Vector Stores API using PGVector",
    version="1.0.0",
)

app.include_router(files.router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

db = database_instance.session()

# async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     """Validate API key from Authorization header"""
#     expected_key = settings.server_api_key
#     if credentials.credentials != expected_key:
#         raise HTTPException(status_code=401, detail="Invalid API key")
#     return credentials.credentials


# @app.on_event("startup")
# async def startup():
#     """Connect to database on startup"""
#     await db.connect()


@app.on_event("shutdown")
async def shutdown():
    """Disconnect from database on shutdown"""
    db.close()


async def generate_query_embedding(query: str, litellm_vkey: str) -> List[float]:
    """
    Generate an embedding for the query using LiteLLM
    """
    return await embedding_service.generate_embedding(query, litellm_vkey)


async def generate_query_embeddings(queries: List[str], litellm_vkey: str) -> List[List[float]]:
    return await embedding_service.generate_embeddings(queries, litellm_vkey)


@app.post("/v1/vector_stores", response_model=VectorStoreResponse)
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

        db.add(store)
        db.commit()

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


@app.get("/v1/vector_stores", response_model=VectorStoreListResponse)
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
            statement = statement.where(col(VectorStore.id) > uuid.UUID(after))

        if before:
            statement = statement.where(col(VectorStore.id) < uuid.UUID(before))

        if team:
            statement = statement.where(VectorStore.team_id == team)
        elif user:
            statement = statement.where(VectorStore.user_id == user)

        statement = statement.order_by(desc(VectorStore.created_at)).limit(limit + 1)

        res = db.exec(statement)
        stores = res.all()

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

        # Determine first_id and last_id
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


@app.post(
    "/v1/vector_stores/{vector_store_id}/search",
    response_model=VectorStoreSearchResponse,
)
@app.post(
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

        statement = select(VectorStore).where(col(VectorStore.id) == uuid.UUID(vector_store_id))
        if team:
            statement = statement.where(VectorStore.team_id == team)
        elif user:
            statement = statement.where(VectorStore.user_id == user)
        else:
            raise HTTPException(status_code=401, detail="No valid credentials provided")

        res = db.exec(statement)
        store = res.first()
        if not store:
            raise HTTPException(status_code=404, detail="Vector store not found")

        # Generate embedding for query
        query_embedding = await generate_query_embedding(request.query, litellm_vkey_info['key'])
        query_embedding = query_embedding + [0] * (3072 - len(query_embedding))

        # Build the raw SQL query for vector similarity search
        limit = min(request.limit or 20, 100)  # Cap at 100 results

        embedding_statement = select(
            Embedding.id, 
            Embedding.content, 
            Embedding.embedding_metadata, 
            Embedding.embedding.l2_distance(query_embedding).label('distance') # pyright: ignore[reportAttributeAccessIssue]
            ).where(col(VectorStore.id) == uuid.UUID(vector_store_id))

        if request.filters:
            for key, value in request.filters.items():
                embedding_statement = embedding_statement.where(col(Embedding.embedding_metadata)[key].as_string() == value)


        embedding_statement = embedding_statement.order_by(label('distance', col(Embedding.embedding)).asc()).limit(limit)
        embedding_results = db.exec(embedding_statement).all()

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


# @app.post(
#     "/v1/vector_stores/{vector_store_id}/embeddings", response_model=EmbeddingResponse
# )
# async def create_embedding(
#     vector_store_id: str,
#     request: EmbeddingCreateRequest,
#     litellm_vkey_info = Depends(get_litellm_vkey_info),
# ):
#     """
#     Add a single embedding to a vector store.
#     """
#     try:
#         # Check if vector store exists
#         # TODO deduplicate
#         team = litellm_vkey_info['info']['team_id']
#         user = None if team else litellm_vkey_info['info']['user_id']

#         statement = select(VectorStore).where(col(VectorStore.id) == uuid.UUID(vector_store_id))
#         if team:
#             statement = statement.where(VectorStore.team_id == team)
#         elif user:
#             statement = statement.where(VectorStore.user_id == user)
#         else:
#             raise HTTPException(status_code=401, detail="No valid credentials provided")
        
#         vector_store = db.exec(statement).first()
#         if not vector_store:
#             raise HTTPException(status_code=404, detail="Vector store not found")

#         new_embedding = Embedding(
#             vector_store_id=uuid.UUID(vector_store_id),
#             content=request.content,
#             embedding=request.embedding + [0] * (3072 - len(request.embedding)),
#             embedding_metadata=request.metadata or {}
#         )

#         db.add(new_embedding)
#         db.commit()

#         if not new_embedding.id:
#             raise HTTPException(status_code=500, detail="Failed to create embedding")

#         # TODO do this through the ORM classes
#         update_vector_store_table_statement = f"""
#             UPDATE vectorstore
#             SET
#                 file_counts = jsonb_set(
#                     jsonb_set(
#                         COALESCE(file_counts, '{{"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}}'::jsonb),
#                         '{{completed}}',
#                         (COALESCE(file_counts->>'completed', '0')::int + 1)::text::jsonb
#                     ),
#                     '{{total}}',
#                     (COALESCE(file_counts->>'total', '0')::int + 1)::text::jsonb
#                 ),
#                 usage_bytes = COALESCE(usage_bytes, 0) + LENGTH(:content),
#                 last_active_at = NOW()
#             WHERE id = :vector_store_id
#             """

#         res = db.connection().execute(text(update_vector_store_table_statement), {'vector_store_id': vector_store_id, 'content': request.content})

#         return EmbeddingResponse(
#             id=new_embedding.id.hex,
#             vector_store_id=new_embedding.vector_store_id.hex,
#             content=new_embedding.content,
#             metadata=new_embedding.embedding_metadata or {},
#             created_at=int(new_embedding.created_at.timestamp()),
#         )

#     except HTTPException:
#         raise
#     except Exception as e:
#         import traceback

#         traceback.print_exc()
#         raise HTTPException(
#             status_code=500, detail=f"Failed to create embedding: {str(e)}"
#         )


# TODO check if this API endpoint is valid and needed, I guess it's superseeded by https://developers.openai.com/api/reference/resources/vector_stores/subresources/file_batches/methods/create
# @app.post(
#     "/v1/vector_stores/{vector_store_id}/embeddings/batch",
#     response_model=EmbeddingBatchCreateResponse,
# )
# async def create_embeddings_batch(
#     vector_store_id: str,
#     request: EmbeddingBatchCreateRequest,
#     litellm_vkey_info = Depends(get_litellm_vkey_info),
# ):
#     return await _create_embeddings_batch(
#         vector_store_id=vector_store_id, embeddings=request, litellm_vkey_info=litellm_vkey_info
#     )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": int(time.time())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
