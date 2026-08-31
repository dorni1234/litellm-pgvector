import os
import asyncio
import time
import shutil
import uuid
import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from prisma import Prisma
from dotenv import load_dotenv
from typing import Annotated
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from httpx import AsyncClient

from models import (
    VectorStoreCreateRequest,
    VectorStoreResponse,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
    SearchResult,
    EmbeddingCreateRequest,
    EmbeddingResponse,
    EmbeddingBatchCreateRequest,
    EmbeddingBatchCreateResponse,
    VectorStoreListResponse,
    ContentChunk,
    VectorStoreFileResponse,
    VectorStoreFileRequest,
    UploadFileRequest,
    UploadFileResponse,
)
from config import settings
from embedding_service import embedding_service
from classes.s3_file_handler import S3FileHandler
from util import get_litellm_vkey_info
from classes.database import database_instance
from routers import files

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

db = database_instance._db

async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key from Authorization header"""
    expected_key = settings.server_api_key
    if credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

@app.on_event("startup")
async def startup():
    """Connect to database on startup"""
    await db.connect()


@app.on_event("shutdown")
async def shutdown():
    """Disconnect from database on shutdown"""
    await db.disconnect()


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
        # Use raw SQL to insert the vector store with configurable table/field names
        vector_store_table = settings.table_names["vector_stores"]
        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']

        result = await db.query_raw(
            f"""
            INSERT INTO {vector_store_table} (id, name, user_id, team_id, file_counts, status, usage_bytes, expires_after, metadata, created_at)
            VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, NOW())
            RETURNING id, name, file_counts, status, usage_bytes, expires_after, expires_at, last_active_at, metadata,
                     EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
            """,
            request.name,
            user,
            team,
            {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            "completed",
            0,
            request.expires_after,
            request.metadata or {},
        )

        print(result)

        if not result:
            raise HTTPException(status_code=500, detail="Failed to create vector store")

        vector_store = result[0]

        # Convert to response format
        created_at = int(vector_store["created_at_timestamp"])
        expires_at = (
            int(vector_store["expires_at"].timestamp())
            if vector_store.get("expires_at")
            else None
        )
        last_active_at = (
            int(vector_store["last_active_at"].timestamp())
            if vector_store.get("last_active_at")
            else None
        )

        return VectorStoreResponse(
            id=vector_store["id"],
            created_at=created_at,
            name=vector_store["name"],
            usage_bytes=vector_store["usage_bytes"] or 0,
            file_counts=vector_store["file_counts"]
            or {
                "in_progress": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 0,
            },
            status=vector_store["status"],
            expires_after=vector_store["expires_after"],
            expires_at=expires_at,
            last_active_at=last_active_at,
            metadata=vector_store["metadata"],
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

        vector_store_table = settings.table_names["vector_stores"]
        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']
        # Build base query
        base_query = f"""
        SELECT id, name, file_counts, status, usage_bytes, expires_after, expires_at, last_active_at, metadata,
               EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
        FROM {vector_store_table}
        """

        # Add pagination conditions
        conditions = []
        params = []
        param_count = 1

        if after:
            conditions.append(f"id > ${param_count}")
            params.append(after)
            param_count += 1

        if before:
            conditions.append(f"id < ${param_count}")
            params.append(before)
            param_count += 1

        if team:
            conditions.append(f"team_id = ${param_count}")
            params.append(team)
            param_count += 1
        elif user:
            conditions.append(f"user_id = ${param_count}")
            params.append(user)
            param_count += 1

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        # Add ordering and limit
        final_query = base_query + f" ORDER BY created_at DESC LIMIT {limit + 1}"

        print(final_query)

        # Execute query
        results = await db.query_raw(final_query, *params)

        # Check if there are more results
        has_more = len(results) > limit
        if has_more:
            results = results[:limit]  # Remove extra result

        # Convert to response format
        vector_stores = []
        for row in results:
            created_at = int(row["created_at_timestamp"])
            expires_at = (
                int(row["expires_at"].timestamp()) if row.get("expires_at") else None
            )
            last_active_at = (
                int(row["last_active_at"].timestamp())
                if row.get("last_active_at")
                else None
            )

            vector_store = VectorStoreResponse(
                id=row["id"],
                created_at=created_at,
                name=row["name"],
                usage_bytes=row["usage_bytes"] or 0,
                file_counts=row["file_counts"]
                or {
                    "in_progress": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "total": 0,
                },
                status=row["status"],
                expires_after=row["expires_after"],
                expires_at=expires_at,
                last_active_at=last_active_at,
                metadata=row["metadata"],
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
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']
        base_query = f"SELECT id FROM {vector_store_table} WHERE id = $1"
        search_param = None
        if team:
            base_query += " AND team_id = $2"
            search_param = team
        elif user:
            base_query += " AND user_id = $2"
            search_param = user
        else:
            raise HTTPException(status_code=401, detail="No valid credentials provided")

        vector_store_result = await db.query_raw(
            base_query, vector_store_id, search_param
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")

        # Generate embedding for query
        query_embedding = await generate_query_embedding(request.query, litellm_vkey_info['key'])
        query_vector_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # Build the raw SQL query for vector similarity search
        limit = min(request.limit or 20, 100)  # Cap at 100 results

        # Base query with vector similarity using cosine distance
        # Use configurable field names
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]

        # Build query with proper parameter placeholders for Prisma
        param_count = 1
        query_params = [query_vector_str, vector_store_id]

        base_query = f"""
        SELECT
            {fields.id_field},
            {fields.content_field},
            {fields.metadata_field},
            ({fields.embedding_field} <=> ${param_count}::vector) as distance
        FROM {table_name}
        WHERE {fields.vector_store_id_field} = ${param_count + 1}
        """
        param_count += 2

        # Add metadata filters if provided
        filter_conditions = []

        if request.filters:
            for key, value in request.filters.items():
                filter_conditions.append(
                    f"{fields.metadata_field}->>${param_count} = ${param_count + 1}"
                )
                query_params.extend([key, str(value)])
                param_count += 2

        if filter_conditions:
            base_query += " AND " + " AND ".join(filter_conditions)

        # Add ordering and limit
        final_query = base_query + f" ORDER BY distance ASC LIMIT {limit}"

        # Execute the query
        results = await db.query_raw(final_query, *query_params)

        # Convert results to SearchResult objects
        search_results = []
        for row in results:
            # Convert distance to similarity score (1 - normalized_distance)
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            similarity_score = max(0, 1 - (row["distance"] / 2))

            # Extract filename from metadata or use a default
            metadata = row[fields.metadata_field] or {}
            filename = metadata.get("filename", "document.txt")

            content_chunks = [ContentChunk(type="text", text=row[fields.content_field])]

            result = SearchResult(
                file_id=row[fields.id_field],
                filename=filename,
                score=similarity_score,
                attributes=metadata if request.return_metadata else None,
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


@app.post(
    "/v1/vector_stores/{vector_store_id}/embeddings", response_model=EmbeddingResponse
)
async def create_embedding(
    vector_store_id: str,
    request: EmbeddingCreateRequest,
    litellm_vkey_info = Depends(get_litellm_vkey_info),
):
    """
    Add a single embedding to a vector store.
    """
    try:
        # Check if vector store exists
        # TODO deduplicate
        vector_store_table = settings.table_names["vector_stores"]
        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']
        base_query = f"SELECT id FROM {vector_store_table} WHERE id = $1"
        search_param = None
        if team:
            base_query += " AND team_id = $2"
            search_param = team
        elif user:
            base_query += " AND user_id = $2"
            search_param = user
        else:
            raise HTTPException(status_code=401, detail="No valid credentials provided")

        vector_store_result = await db.query_raw(
            base_query, vector_store_id, search_param
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")

        # Convert embedding to vector string format
        embedding_vector_str = "[" + ",".join(map(str, request.embedding)) + "]"

        # Insert embedding using configurable field names
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]

        result = await db.query_raw(
            f"""
            INSERT INTO {table_name} ({fields.id_field}, {fields.vector_store_id_field}, {fields.content_field},
                                     {fields.embedding_field}, {fields.metadata_field}, {fields.created_at_field})
            VALUES (gen_random_uuid(), $1, $2, $3::vector, $4, NOW())
            RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.content_field},
                     {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            """,
            vector_store_id,
            request.content,
            embedding_vector_str,
            request.metadata or {},
        )

        if not result:
            raise HTTPException(status_code=500, detail="Failed to create embedding")

        embedding = result[0]

        # Update vector store statistics
        await db.query_raw(
            f"""
            UPDATE {vector_store_table}
            SET
                file_counts = jsonb_set(
                    jsonb_set(
                        COALESCE(file_counts, '{{"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}}'::jsonb),
                        '{{completed}}',
                        (COALESCE(file_counts->>'completed', '0')::int + 1)::text::jsonb
                    ),
                    '{{total}}',
                    (COALESCE(file_counts->>'total', '0')::int + 1)::text::jsonb
                ),
                usage_bytes = COALESCE(usage_bytes, 0) + LENGTH($2),
                last_active_at = NOW()
            WHERE id = $1
            """,
            vector_store_id,
            request.content,
        )

        return EmbeddingResponse(
            id=embedding[fields.id_field],
            vector_store_id=embedding[fields.vector_store_id_field],
            content=embedding[fields.content_field],
            metadata=embedding[fields.metadata_field],
            created_at=int(embedding["created_at_timestamp"]),
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to create embedding: {str(e)}"
        )


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


async def _create_embeddings_batch(
    vector_store_id: str, vector_store_file_id: str, embeddings: List[EmbeddingCreateRequest], litellm_vkey_info
):
    """
    Add multiple embeddings to a vector store in batch.
    """
    try:
        # TODO deduplicate
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        team = litellm_vkey_info['info']['team_id']
        user = None if team else litellm_vkey_info['info']['user_id']
        base_query = f"SELECT id FROM {vector_store_table} WHERE id = $1"
        search_param = None
        if team:
            base_query += " AND team_id = $2"
            search_param = team
        elif user:
            base_query += " AND user_id = $2"
            search_param = user
        else:
            raise HTTPException(status_code=401, detail="No valid credentials provided")

        vector_store_result = await db.query_raw(
            base_query, vector_store_id, search_param
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")

        if not embeddings:
            raise HTTPException(status_code=400, detail="No embeddings provided")

        # Prepare batch insert
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]

        # Build VALUES clause for batch insert
        values_clauses = []
        params = []
        param_count = 1

        for embedding_req in embeddings:
            embedding_vector_str = (
                "[" + ",".join(map(str, embedding_req.embedding)) + "]"
            )
            values_clauses.append(
                f"(gen_random_uuid(), ${param_count}, ${param_count + 1}, ${param_count + 2}, ${param_count + 3}::vector, ${param_count + 4}, NOW())"
            )
            params.extend(
                [
                    vector_store_id,
                    vector_store_file_id, 
                    embedding_req.content or {},
                    embedding_vector_str,
                    embedding_req.metadata or {},
                ]
            )
            param_count += 5

        values_clause = ", ".join(values_clauses)

        # Execute batch insert
        result = await db.query_raw(
            f"""
            INSERT INTO {table_name} ({fields.id_field}, {fields.vector_store_id_field}, {fields.vector_store_file_id_field}, {fields.content_field},
                                     {fields.embedding_field}, {fields.metadata_field}, {fields.created_at_field})
            VALUES {values_clause}
            RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.vector_store_file_id_field}, {fields.content_field},
                     {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            """,
            *params,
        )

        if not result:
            raise HTTPException(status_code=500, detail="Failed to create embeddings")

        # Calculate total content length for usage bytes update
        total_content_length = sum(len(emb.content) for emb in embeddings)

        # Update vector store statistics
        await db.query_raw(
            f"""
            UPDATE {vector_store_table}
            SET
                file_counts = jsonb_set(
                    jsonb_set(
                        COALESCE(file_counts, '{{"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}}'::jsonb),
                        '{{completed}}',
                        (COALESCE(file_counts->>'completed', '0')::int + $2)::text::jsonb
                    ),
                    '{{total}}',
                    (COALESCE(file_counts->>'total', '0')::int + $2)::text::jsonb
                ),
                usage_bytes = COALESCE(usage_bytes, 0) + $3,
                last_active_at = NOW()
            WHERE id = $1
            """,
            vector_store_id,
            len(embeddings),
            total_content_length,
        )

        # Convert results to response format
        result_embeddings = []
        for row in result:
            result_embeddings.append(
                EmbeddingResponse(
                    id=row[fields.id_field],
                    vector_store_id=row[fields.vector_store_id_field],
                    content=row[fields.content_field],
                    metadata=row[fields.metadata_field],
                    created_at=int(row["created_at_timestamp"]),
                )
            )

        return EmbeddingBatchCreateResponse(
            data=result_embeddings,
            created=int(time.time()),
            total_content_length=total_content_length,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to create embeddings batch: {str(e)}"
        )


@app.post(
    "/v1/vector_stores/{vector_store_id}/files",
    response_model=VectorStoreFileResponse,
)
async def create_vector_store_file(
    vector_store_id: str,
    request: VectorStoreFileRequest,
    litellm_vkey_info = Depends(get_litellm_vkey_info),
):
    """
    Compute embeddings for prevoiusly uploaded file and insert them into the vector store.
    """
    if litellm_vkey_info['info']['team_id']:
        file_object = await db.file.find_unique(where={"id": request.file_id, "team_id": litellm_vkey_info['info']['team_id']})
    elif litellm_vkey_info['info']['user_id']:
        file_object = await db.file.find_unique(where={"id": request.file_id, "user_id": litellm_vkey_info['info']['user_id']})
        print(request.file_id)
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if file_object is None:
        raise HTTPException(status_code=404, detail="Invalid file id")

    litellm_vkey = litellm_vkey_info['key']

    s3_file_handler = S3FileHandler(
        region=settings.s3_region,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        bucket=settings.s3_bucket,
        host=settings.s3_host
        )
    
    filepath = await s3_file_handler.get_file(file_object.filename_on_disk)
    converter = DocumentConverter()
    result = converter.convert(filepath)
    doc = result.document
    os.remove(filepath)

    # TODO check if this is a good model for tokenization
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        max_tokens=25,
    )

    chunker = LineBasedTokenChunker(
        tokenizer=tokenizer,
        prefix="",  # No prefix for general documents
    )

    chunks = list(chunker.chunk(doc))
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk.text)

    embedding_vectors = await generate_query_embeddings(chunk_texts, litellm_vkey)

    embedding_create_requests = []

    for index, chunk_text in enumerate(chunk_texts):
        embedding_create_requests.append(
            EmbeddingCreateRequest(
                content=chunk_text,
                embedding=embedding_vectors[index]["embedding"],
            )
        )

    dbobject = await db.vectorstorefile.create(
        data={
            "file_id": request.file_id,
            "user_id": litellm_vkey_info['info']['user_id'] if not litellm_vkey_info['info']['team_id'] else None,
            "team_id": litellm_vkey_info['info']['team_id'],
            "vector_store_id": vector_store_id,
            "attributes": None,
            "chunking_strategy": None,
            "created_at": datetime.datetime.now(),
            "usage_bytes": 0,
        }
    )

    embedding_result = await _create_embeddings_batch(
        vector_store_id=vector_store_id, vector_store_file_id=dbobject.id, embeddings=embedding_create_requests, litellm_vkey_info=litellm_vkey_info
    )

    await db.vectorstorefile.update(
        where={'id': dbobject.id},
        data={'usage_bytes': embedding_result.total_content_length}
    )

    return VectorStoreFileResponse(
        id=dbobject.id,
        created_at=dbobject.created_at.strftime("%s"),
        object="vector_store.file",
        status="completed",
        usage_bytes=embedding_result.total_content_length,
        vector_store_id=vector_store_id,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": int(time.time())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
