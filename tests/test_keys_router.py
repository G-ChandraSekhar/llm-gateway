import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_key_returns_raw_key_once(client: AsyncClient):
    resp = await client.post("/v1/keys", json={"name": "test key"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "test key"
    assert body["api_key"].startswith("sk-gw-")
    assert body["prefix"] == body["api_key"][:12]


@pytest.mark.asyncio
async def test_whoami_with_valid_key_succeeds(client: AsyncClient):
    create_resp = await client.post("/v1/keys", json={"name": "alice", "budget_limit_usd": 5.0})
    raw_key = create_resp.json()["api_key"]

    resp = await client.get("/v1/keys/me", headers={"Authorization": f"Bearer {raw_key}"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "alice"
    assert body["is_active"] is True
    assert body["budget_limit_usd"] == 5.0
    assert body["spent_usd"] == 0.0


@pytest.mark.asyncio
async def test_whoami_without_key_is_401(client: AsyncClient):
    resp = await client.get("/v1/keys/me")

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Missing API key"


@pytest.mark.asyncio
async def test_whoami_with_garbage_key_is_401(client: AsyncClient):
    resp = await client.get("/v1/keys/me", headers={"Authorization": "Bearer not-a-real-key"})

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid or inactive API key"


@pytest.mark.asyncio
async def test_whoami_with_one_keys_raw_value_does_not_authenticate_as_another(client: AsyncClient):
    # Two distinct keys must never cross-authenticate — sanity check that
    # lookup is by exact hash match, not e.g. prefix.
    resp1 = await client.post("/v1/keys", json={"name": "key one"})
    await client.post("/v1/keys", json={"name": "key two"})

    key1 = resp1.json()["api_key"]
    resp = await client.get("/v1/keys/me", headers={"Authorization": f"Bearer {key1}"})

    assert resp.status_code == 200
    assert resp.json()["name"] == "key one"
