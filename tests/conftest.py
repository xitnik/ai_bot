import asyncio
import os
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

try:  # SQLAlchemy <2.0 совместимость
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
except ImportError:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker as async_sessionmaker


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import procurement.db as db  # noqa: E402
from app import app  # noqa: E402


@pytest_asyncio.fixture(scope="session")
def event_loop() -> AsyncGenerator[asyncio.AbstractEventLoop, None]:
    # Отдельный event loop на сессию, чтобы async фикстуры работали корректно.
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture()
async def test_db(tmp_path_factory):
    db_dir = tmp_path_factory.mktemp("data")
    url = f"sqlite+aiosqlite:///{db_dir}/test.db"
    os.environ["DATABASE_URL"] = url

    engine = db.create_engine(url)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    db.engine = engine
    db.AsyncSessionMaker = session_maker
    await db.init_db(engine)

    yield session_maker

    await engine.dispose()


@pytest_asyncio.fixture()
async def client(test_db) -> AsyncGenerator[tuple[AsyncClient, async_sessionmaker], None]:
    session_maker: async_sessionmaker = test_db

    async def override_session():
        async with session_maker() as session:
            yield session

    app.dependency_overrides[db.get_session] = override_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client, session_maker

    app.dependency_overrides.clear()
