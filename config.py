from typing import Dict, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from urllib.parse import urlparse, parse_qs

def get_db_schema_name(uri) -> str:
    parsed_uri = urlparse(uri)
    if not parsed_uri.query:
        raise Exception('No schema provided in the DATABASE_URL uri')
    parsed_query_string = parse_qs(parsed_uri.query)
    if not 'schema' in parsed_query_string:
        raise Exception('No schema provided in the DATABASE_URL uri')

    return str(parsed_query_string['schema'].pop())

class DatabaseFieldConfig(BaseModel):
    """Configuration for database field mappings"""
    id_field: str = "id"
    content_field: str = "content"
    metadata_field: str = "metadata"
    embedding_field: str = "embedding"
    vector_store_id_field: str = "vector_store_id"
    vector_store_file_id_field: str = "vector_store_file_id"
    created_at_field: str = "created_at"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation via LiteLLM proxy"""
    model: str = "text-embedding-ada-002"
    base_url: str = "http://localhost:4000"  # LiteLLM proxy URL
    api_key: str = "sk-1234"  # LiteLLM proxy API key
    dimensions: int = 1536


class Settings(BaseSettings):
    """Application settings"""
    # Database configuration
    database_url: str = ""
    database_url_2: str = ""

    # API configuration
    server_api_key: str = "your-api-key-here"
    port: int = 8000
    host: str = "0.0.0.0"

    # Database field mappings
    db_fields: DatabaseFieldConfig = DatabaseFieldConfig()

    # Embedding configuration
    embedding: EmbeddingConfig = EmbeddingConfig()

    # S3 config
    s3_host: str = ""
    s3_region: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = ""

    # LiteLLM config
    litellm_api_key: str = ""

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False

        # Allow environment variables like:
        # DB_FIELDS__ID_FIELD=custom_id
        # EMBEDDING__MODEL=text-embedding-3-small
        # EMBEDDING__API_BASE=https://api.openai.com/v1

    @property
    def table_names(self) -> Dict[str, str]:
        """Get table names"""
        return {
            "vector_stores": "vector_stores",
            "embeddings": "embeddings",
            "vector_store_files": "VectorStoreFiles",
            "files": "File"
        }

    @property
    def database_schema(self) -> str:
        return get_db_schema_name(self.database_url_2)

# Global settings instance
settings = Settings()
