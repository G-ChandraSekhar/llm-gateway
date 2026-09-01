import pytest
from httpx import AsyncClient

from tests.conftest import ADMIN_HEADERS


@pytest.mark.asyncio
async def test_create_key_returns_raw_key_once(client: AsyncClient):
    resp = await client.post("/v1/keys", json={"name": "test key"}, headers=ADMIN_HEADERS)

    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "test key"
    assert body["api_key"].startswith("sk-gw-")
    assert body["prefix"] == body["api_key"][:12]


@pytest.mark.asyncio
async def test_create_key_without_admin_auth_is_rejected(client: AsyncClient):
    resp = await client.post("/v1/keys", json={"name": "no admin header"})

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Missing admin credentials"


@pytest.mark.asyncio
async def test_create_key_with_wrong_admin_secret_is_rejected(client: AsyncClient):
    resp = await client.post(
        "/v1/keys", json={"name": "wrong secret"}, headers={"Authorization": "Bearer not-the-real-admin-key"}
    )

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid admin credentials"


@pytest.mark.asyncio
async def test_whoami_with_valid_key_succeeds(client: AsyncClient):
    create_resp = await client.post(
        "/v1/keys", json={"name": "alice", "budget_limit_usd": 5.0}, headers=ADMIN_HEADERS
    )
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
    resp1 = await client.post("/v1/keys", json={"name": "key one"}, headers=ADMIN_HEADERS)
    await client.post("/v1/keys", json={"name": "key two"}, headers=ADMIN_HEADERS)

    key1 = resp1.json()["api_key"]
    resp = await client.get("/v1/keys/me", headers={"Authorization": f"Bearer {key1}"})

    assert resp.status_code == 200
    assert resp.json()["name"] == "key one"


@pytest.mark.asyncio
async def test_list_keys_requires_admin(client: AsyncClient):
    resp = await client.get("/v1/keys")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_list_keys_returns_all_created_keys(client: AsyncClient):
    await client.post("/v1/keys", json={"name": "key one"}, headers=ADMIN_HEADERS)
    await client.post("/v1/keys", json={"name": "key two"}, headers=ADMIN_HEADERS)

    resp = await client.get("/v1/keys", headers=ADMIN_HEADERS)

    assert resp.status_code == 200
    names = {k["name"] for k in resp.json()}
    assert names == {"key one", "key two"}
    # Raw key material must never appear in the listing.
    assert all("api_key" not in k for k in resp.json())


@pytest.mark.asyncio
async def test_revoke_key_requires_admin(client: AsyncClient):
    create_resp = await client.post("/v1/keys", json={"name": "target"}, headers=ADMIN_HEADERS)
    key_id = create_resp.json()["id"]

    resp = await client.delete(f"/v1/keys/{key_id}")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_revoke_key_deactivates_it(client: AsyncClient):
    create_resp = await client.post("/v1/keys", json={"name": "target"}, headers=ADMIN_HEADERS)
    key_id = create_resp.json()["id"]
    raw_key = create_resp.json()["api_key"]

    resp = await client.delete(f"/v1/keys/{key_id}", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False

    # The revoked key must immediately fail auth on its own behalf.
    whoami_resp = await client.get("/v1/keys/me", headers={"Authorization": f"Bearer {raw_key}"})
    assert whoami_resp.status_code == 401
    assert whoami_resp.json()["detail"] == "Invalid or inactive API key"


@pytest.mark.asyncio
async def test_revoke_nonexistent_key_is_404(client: AsyncClient):
    resp = await client.delete("/v1/keys/00000000-0000-0000-0000-000000000000", headers=ADMIN_HEADERS)
    assert resp.status_code == 404
