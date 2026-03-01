"""Unit tests for secrets.py — encrypt/decrypt, keyfile management."""

import os
import stat
from unittest.mock import patch

import pytest

from ollamacode.secrets import (
    _ensure_keyfile,
    _get_cipher_key,
    decrypt,
    delete_secret,
    encrypt,
    get_secret,
    list_secrets,
    resolve_secret,
    set_secret,
)


# ---------------------------------------------------------------------------
# Keyfile management
# ---------------------------------------------------------------------------


class TestKeyfile:
    def test_ensure_keyfile_creates_on_first_run(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        key = _ensure_keyfile()
        assert len(key) == 32
        assert keyfile.exists()
        mode = stat.S_IMODE(keyfile.stat().st_mode)
        assert mode == 0o600

    def test_ensure_keyfile_returns_existing(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        expected_key = os.urandom(32)
        keyfile.write_bytes(expected_key)
        keyfile.chmod(0o600)
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        key = _ensure_keyfile()
        assert key == expected_key

    def test_ensure_keyfile_warns_bad_permissions(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        keyfile.write_bytes(os.urandom(32))
        keyfile.chmod(0o644)  # Wrong permissions
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        with pytest.warns(UserWarning, match="0o644"):
            _ensure_keyfile()

    def test_get_cipher_key_returns_32_bytes(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        keyfile.write_bytes(os.urandom(64))  # Longer than needed
        keyfile.chmod(0o600)
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        key = _get_cipher_key()
        assert len(key) == 32

    def test_get_cipher_key_hashes_short_keyfile(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        keyfile.write_bytes(b"short")  # Less than 32 bytes
        keyfile.chmod(0o600)
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        key = _get_cipher_key()
        assert len(key) == 32  # SHA-256 digest is always 32 bytes


# ---------------------------------------------------------------------------
# Encrypt / Decrypt roundtrip
# ---------------------------------------------------------------------------


class TestEncryptDecrypt:
    def test_roundtrip(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        plaintext = "my-secret-api-key-12345"
        ciphertext = encrypt(plaintext)
        assert ciphertext.startswith("v1:")
        assert plaintext not in ciphertext

        decrypted = decrypt(ciphertext)
        assert decrypted == plaintext

    def test_different_encryptions_produce_different_ciphertext(
        self, tmp_path, monkeypatch
    ):
        keyfile = tmp_path / "keyfile"
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        ct1 = encrypt("same-value")
        ct2 = encrypt("same-value")
        # Different nonces should produce different ciphertext
        assert ct1 != ct2

    def test_decrypt_bad_format_raises(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        keyfile.write_bytes(os.urandom(32))
        keyfile.chmod(0o600)
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        with pytest.raises(ValueError, match="Unrecognized ciphertext format"):
            decrypt("bad-format:data")

    def test_decrypt_too_short_raises(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        keyfile.write_bytes(os.urandom(32))
        keyfile.chmod(0o600)
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        import base64

        short = base64.b64encode(b"short").decode()
        with pytest.raises(ValueError, match="too short"):
            decrypt(f"v1:{short}")

    def test_decrypt_tampered_data_raises(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        ct = encrypt("secret")
        # Tamper with the ciphertext
        import base64

        raw = bytearray(base64.b64decode(ct[3:]))
        raw[-1] ^= 0xFF  # Flip a byte
        tampered = "v1:" + base64.b64encode(bytes(raw)).decode()
        with pytest.raises(ValueError, match="Decryption failed"):
            decrypt(tampered)


# ---------------------------------------------------------------------------
# resolve_secret
# ---------------------------------------------------------------------------


class TestResolveSecret:
    def test_non_secret_value_passed_through(self):
        assert resolve_secret("plain-value") == "plain-value"
        assert resolve_secret(None) is None
        assert resolve_secret("") == ""

    def test_secret_prefix_resolves(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        # Mock the state module's _load_raw (imported locally inside secrets functions)
        ct = encrypt("my-api-key")
        mock_data = {"encrypted_secrets": {"test_key": ct}}

        with patch("ollamacode.state._load_raw", return_value=mock_data):
            result = resolve_secret("secret:test_key")
        assert result == "my-api-key"

    def test_secret_not_found_raises_key_error(self):
        mock_data = {"encrypted_secrets": {}}
        with patch("ollamacode.state._load_raw", return_value=mock_data):
            with pytest.raises(KeyError, match="not found"):
                resolve_secret("secret:nonexistent")


# ---------------------------------------------------------------------------
# High-level store operations (set/get/delete/list)
# ---------------------------------------------------------------------------


class TestSecretStore:
    def test_set_and_get_secret(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        store = {}

        def mock_load_raw():
            return dict(store)

        def mock_save(data):
            store.clear()
            store.update(data)

        with patch("ollamacode.state._load_raw", side_effect=mock_load_raw):
            with patch("ollamacode.state._save", side_effect=mock_save):
                set_secret("my_key", "my_value")
                result = get_secret("my_key")
        assert result == "my_value"

    def test_delete_secret(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        ct = encrypt("val")
        store = {"encrypted_secrets": {"key1": ct}}

        def mock_load_raw():
            return dict(store)

        def mock_save(data):
            store.clear()
            store.update(data)

        with patch("ollamacode.state._load_raw", side_effect=mock_load_raw):
            with patch("ollamacode.state._save", side_effect=mock_save):
                assert delete_secret("key1") is True
                assert delete_secret("nonexistent") is False

    def test_list_secrets(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        store = {"encrypted_secrets": {"beta": "v1:x", "alpha": "v1:y"}}
        with patch("ollamacode.state._load_raw", return_value=store):
            names = list_secrets()
        assert names == ["alpha", "beta"]

    def test_set_secret_empty_name_raises(self, tmp_path, monkeypatch):
        keyfile = tmp_path / "keyfile"
        monkeypatch.setattr("ollamacode.secrets._KEYFILE_PATH", keyfile)
        monkeypatch.setattr("ollamacode.secrets._OLLAMACODE_DIR", tmp_path)

        with pytest.raises(ValueError, match="empty"):
            set_secret("", "value")

    def test_get_secret_missing_returns_none(self):
        store = {"encrypted_secrets": {}}
        with patch("ollamacode.state._load_raw", return_value=store):
            assert get_secret("nonexistent") is None
