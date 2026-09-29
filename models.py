from typing import Optional, Dict, Any, List, Union, Annotated
from pydantic import BaseModel
from datetime import datetime
from typing_extensions import Literal, TypeAlias
from fastapi import UploadFile, Form


class VectorStoreCreateRequest(BaseModel):
    name: str
    file_ids: Optional[List[str]] = None
    expires_after: Optional["VectorStoreExpiresAfterObject"] = None
    chunking_strategy: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorStoreResponse(BaseModel):
    id: str
    object: str = "vector_store"
    created_at: int
    name: str
    usage_bytes: int
    file_counts: Dict[str, int]
    status: str
    expires_after: Optional["VectorStoreExpiresAfterObject"] = None
    expires_at: Optional[int] = None
    last_active_at: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorStoreSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 20
    filters: Optional[Dict[str, Any]] = None
    return_metadata: Optional[bool] = True


class ContentChunk(BaseModel):
    type: str = "text"
    text: str


class SearchResult(BaseModel):
    file_id: str
    filename: str
    score: float
    attributes: Optional[Dict[str, Any]] = None
    content: List[ContentChunk]


class VectorStoreSearchResponse(BaseModel):
    object: str = "vector_store.search_results.page"
    search_query: str
    data: List[SearchResult]
    has_more: bool = False
    next_page: Optional[str] = None


class EmbeddingCreateRequest(BaseModel):
    content: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingResponse(BaseModel):
    id: str
    object: str = "embedding"
    vector_store_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: int


class EmbeddingBatchCreateRequest(BaseModel):
    embeddings: List[EmbeddingCreateRequest]


class EmbeddingBatchCreateResponse(BaseModel):
    object: str = "embedding.batch"
    data: List[EmbeddingResponse]
    created: int
    total_content_length: int


class VectorStoreListResponse(BaseModel):
    object: str = "list"
    data: List[VectorStoreResponse]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class StaticFileChunkingStrategy(BaseModel):
    chunk_overlap_tokens: int
    max_chunk_size_tokens: int


class AutoFileChunkingStrategyParam(BaseModel):
    type: Literal["auto"]


class StaticFileChunkingStrategyObjectParam(BaseModel):
    type: Literal["auto"]
    static: StaticFileChunkingStrategy


FileChunkingStrategyParam: TypeAlias = Union[
    AutoFileChunkingStrategyParam, StaticFileChunkingStrategyObjectParam
]


class VectorStoreFileRequest(BaseModel):
    file_id: str
    attributes: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[FileChunkingStrategyParam] = None


class ExpiresAfterObject(BaseModel):
    anchor: Literal["created_at"]
    seconds: int


class UploadFileRequest(BaseModel):
    file: UploadFile
    purpose: str
    expires_after: Optional[ExpiresAfterObject] = None


class UploadFileResponse(BaseModel):
    id: str
    object: Literal["file"]
    bytes: int
    created_at: int
    filename: str
    purpose: str
    expires_at: Optional[int] = None


class VectorStoreFileResponse(BaseModel):
    id: str
    created_at: int
    last_error: Optional[Dict[str, Any]] = None
    object: str = "vector_store.file"
    status: str
    usage_bytes: int
    vector_store_id: str
    attributes: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[StaticFileChunkingStrategy] = None

class DeleteFileResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool = True

class VectorStoreRetrieveResponse(BaseModel):
    id: str
    object: str = "vector_store"
    created_at: int
    name: str
    file_counts: Optional[Dict[str, int]] = None
    usage_bytes: int
    last_active_at: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class VectorStoreDeleteResponse(BaseModel):
    id: str
    object: str = "vector_store.deleted"
    deleted: bool = True

class VectorStoreExpiresAfterObject(BaseModel):
    anchor: Literal["last_active_at"]
    days: int

VectorStoreExpiresAfterObject.model_rebuild()

class VectorStoreUpdateRequest(BaseModel):
    expires_after: Optional[VectorStoreExpiresAfterObject] = None
    metadata: Optional[Dict[str, Any]] = None
    name: Optional[str] = None

class VectorStoreUpdateResponse(BaseModel):
    id: str
    object: str = "vector_store"
    created_at: int
    name: str
    file_counts: Optional[Dict[str, int]] = None
    usage_bytes: int
    last_active_at: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None