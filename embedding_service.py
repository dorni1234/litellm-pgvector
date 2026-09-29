from typing import List, Optional
from fastapi import HTTPException
from config import settings, EmbeddingConfig
from litellm.types.utils import EmbeddingResponse as LiteLLMEmbeddingResponse
from uuid import UUID
from time import time
import litellm
import logging

from models import EmbeddingCreateRequest, EmbeddingBatchCreateResponse, EmbeddingResponse
from sqlmodel import select, col, text
from sqlalchemy import label

from classes.database import VectorStore, VectorStoreFile, Embedding, database_instance

class EmbeddingService:
    """Service for generating embeddings using OpenAI SDK pointed at LiteLLM proxy"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or settings.embedding

    async def generate_embedding(self, text: str, litellm_vkey: str) -> List[float]:
        """
        Generate embedding for a single text using LiteLLM proxy

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response: LiteLLMEmbeddingResponse = await litellm.aembedding(
                # TODO implement model selection by request
                model=self.config.model,
                input=[text],
                api_base=self.config.base_url,
                api_key=litellm_vkey,
                dimensions=self.config.dimensions
            )
            logging.debug(f"Embedding response: {response}")

            # Extract embedding from response
            embedding = response.data[0]["embedding"]

            # Validate embedding dimensions
            if len(embedding) != self.config.dimensions:
                raise ValueError(
                    f"Expected embedding dimension {self.config.dimensions}, "
                    f"got {len(embedding)}"
                )

            return embedding

        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    async def generate_embeddings(self, texts: List[str], litellm_vkey: str) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            # Generate embeddings using LiteLLM
            response = await litellm.aembedding(
                # TODO implement model selection by request
                model=self.config.model,
                input=texts,
                api_base=self.config.base_url,
                api_key=litellm_vkey,
                dimensions=self.config.dimensions
            )

            embedding_list = []
            # Validate embedding dimensions
            for i, embedding in enumerate(response.data):
                embedding_data = embedding['embedding']
                if len(embedding_data) != self.config.dimensions:
                    raise ValueError(
                        f"Expected embedding dimension {self.config.dimensions} for text {i}, "
                        f"got {len(embedding_data)}"
                    )
                embedding_list.append(embedding_data)

            return embedding_list

        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

    def update_config(self, new_config: EmbeddingConfig):
        """Update the embedding configuration"""
        self.config = new_config

    async def insert_embeddings(
            self, 
            vector_store_id: str, 
            vector_store_file_id: str, 
            embeddings: List[EmbeddingCreateRequest], 
            litellm_vkey_info
        ):
        """
        Add multiple embeddings to a vector store in batch.
        """
        try:
            # TODO deduplicate
            # Check if vector store exists
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
            
            vector_store = session.exec(statement).first()
            if not vector_store:
                raise HTTPException(status_code=404, detail="Vector store not found")

            if not embeddings:
                raise HTTPException(status_code=400, detail="No embeddings provided")

            new_embeddings = []
            total_content_length = 0
            for embedding_req in embeddings:
                new_embedding = Embedding(
                    vector_store_id=UUID(vector_store_id),
                    vector_store_file_id=UUID(vector_store_file_id),
                    content=embedding_req.content,
                    embedding=embedding_req.embedding + [0] * (3072 - len(embedding_req.embedding)),
                    embedding_metadata=embedding_req.metadata or {}
                )
                new_embeddings.append(new_embedding)
                total_content_length += len(new_embedding.content)

            session.add_all(new_embeddings)
            session.commit()

            # Update vector store statistics
            # TODO do this through the ORM classes
            # TODO add counter to in_progress when adding the job to the job queue
            update_statistics_statement = f"""
                UPDATE {settings.database_schema}.vectorstore
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
                    usage_bytes = COALESCE(usage_bytes, 0) + :total_content_length,
                    last_active_at = NOW()
                WHERE id = :vector_store_id
                """
            
            res = session.connection().execute(
                text(update_statistics_statement),
                {
                    'vector_store_id': vector_store_id,
                    'total_content_length': total_content_length
                }
            )

            # Convert results to response format
            result_embeddings = []
            new_embedding: Embedding
            for new_embedding in new_embeddings:
                result_embeddings.append(
                    EmbeddingResponse(
                        id=new_embedding.id.hex,
                        vector_store_id=new_embedding.vector_store_id.hex,
                        content=new_embedding.content,
                        metadata=new_embedding.embedding_metadata,
                        created_at=int(new_embedding.created_at.timestamp()),
                    )
                )

            return EmbeddingBatchCreateResponse(
                data=result_embeddings,
                created=int(time()),
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

# Global embedding service instance
embedding_service = EmbeddingService()
