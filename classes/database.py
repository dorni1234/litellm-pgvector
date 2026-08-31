from prisma import Prisma
from config import settings

class Database:
    _db: Prisma
    _is_connected: bool = False

    def __init__(self):
        self._db = Prisma()

    async def connect(self):
        if not self._is_connected:
            await self._db.connect()
            self._is_connected = True

    async def disconnect(self):
        if self._is_connected:
            await self._db.disconnect()
            self._is_connected = False

database_instance = Database()