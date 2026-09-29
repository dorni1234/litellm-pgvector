from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from config import settings
from httpx import AsyncClient

security = HTTPBearer()

async def get_litellm_vkey_info(credentials: HTTPAuthorizationCredentials = Depends(security)):
    litellm_api_key = settings.litellm_api_key
    litellm_host = settings.embedding.base_url
    provided_litellm_vkey = credentials.credentials
    async with AsyncClient() as client:
        response = await client.get(
            f"{litellm_host}/key/info?key={provided_litellm_vkey}",
            headers={"Authorization": f"Bearer {litellm_api_key}"}
            )
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return response.json()
