from config import settings

from sqlmodel import Field, SQLModel, create_engine, Session, text, MetaData, Relationship
from sqlalchemy import Column, DATETIME, Engine
from sqlalchemy.schema import CreateSchema
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from pgvector.sqlalchemy import Vector
from datetime import datetime
from pydantic import ConfigDict

from urllib.parse import quote_plus

import uuid

SQLModel.metadata.schema = settings.database_schema

class VectorStore(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str
    user_id: str | None = Field(nullable=True)
    team_id: str | None = Field(nullable=True)
    file_counts: dict | None = Field(sa_column=Column(JSONB))
    status: str = Field(default="completed")
    usage_bytes: int | None = Field(default=0)
    created_at: datetime = Field(sa_column=Column(TIMESTAMP), default_factory=lambda: datetime.now())
    expires_after: dict | None = Field(sa_column=Column(JSONB, nullable=True), default=None)
    expires_at: datetime | None = Field(sa_column=Column(TIMESTAMP, nullable=True), default=None)
    last_active_at: datetime | None = Field(sa_column=Column(TIMESTAMP, nullable=True), default=None)
    store_metadata: dict | None = Field(sa_column=Column(JSONB, nullable=True), default=None)     # orig name was metadata, changed as metadata is reserved in SQLAlchemy

class Embedding(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    vector_store_id: uuid.UUID = Field(default_factory=uuid.uuid4, foreign_key="vectorstore.id", ondelete="CASCADE")
    vector_store_file_id: uuid.UUID = Field(default_factory=uuid.uuid4, foreign_key="vectorstorefile.id", ondelete="CASCADE")
    content: str
    embedding: list[float] = Field(sa_column=Column(Vector(3072)))
    embedding_metadata: dict | None = Field(sa_column=Column(JSONB))     # orig name was metadata, changed as metadata is reserved in SQLAlchemy
    created_at: datetime = Field(sa_column=Column(TIMESTAMP), default_factory=lambda: datetime.now())
    # vector_store_file: VectorStoreFile = Relationship(cascade_delete=True)

class VectorStoreFile(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    file_id: uuid.UUID = Field(foreign_key="file.id", ondelete="CASCADE")
    user_id: str | None = Field(nullable=True)
    team_id: str | None = Field(nullable=True)
    vector_store_id: uuid.UUID = Field(foreign_key="vectorstore.id", ondelete="CASCADE")
    attributes: dict | None = Field(sa_column=Column(JSONB))
    chunking_strategy: str
    usage_bytes: int
    created_at: datetime = Field(sa_column=Column(TIMESTAMP), default_factory=lambda: datetime.now())
    embeddings: list[Embedding] = Relationship(cascade_delete=True)

class File(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: str | None = Field(nullable=True)
    team_id: str | None = Field(nullable=True)
    filename: str
    size: int
    purpose: str
    created_at: datetime = Field(sa_column=Column(TIMESTAMP), default_factory=lambda: datetime.now())
    expires_at: datetime | None = Field(sa_column=Column(TIMESTAMP, nullable=True), default=None)
    filename_on_disk: str
    vector_store_file: VectorStoreFile = Relationship(cascade_delete=True)


class Database:
    _engine: Engine

    def __init__(self):
        # self._engine = create_engine(settings.database_url_2)
        # psycopg2 tends to choke on = signs in the database uri, so we replace it
        self._engine = create_engine(quote_plus(settings.database_url_2, safe=":/?_@"))
        with self._engine.connect() as conn:
            if not conn.dialect.has_schema(conn, settings.database_schema):
                conn.execute(CreateSchema(settings.database_schema))
                conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
                conn.commit()

        SQLModel.metadata.create_all(self._engine)

    def session(self):
        return Session(self._engine)

database_instance = Database()