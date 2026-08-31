from app.core.security import generate_api_key, hash_api_key, key_prefix


def test_generate_api_key_has_prefix_and_is_unique():
    key1 = generate_api_key()
    key2 = generate_api_key()

    assert key1.startswith("sk-gw-")
    assert key1 != key2


def test_hash_api_key_is_deterministic():
    key = generate_api_key()

    assert hash_api_key(key) == hash_api_key(key)


def test_hash_api_key_differs_for_different_keys():
    assert hash_api_key(generate_api_key()) != hash_api_key(generate_api_key())


def test_key_prefix_is_first_12_chars():
    key = "sk-gw-abcdefghijklmnopqrstuvwxyz"

    assert key_prefix(key) == "sk-gw-abcdef"
    assert len(key_prefix(key)) == 12
