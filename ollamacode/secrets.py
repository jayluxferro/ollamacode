"""Encrypted secrets storage for OllamaCode.

API keys and sensitive values are stored encrypted with AES-256-GCM in
~/.ollamacode/state.json under the `encrypted_secrets` key.

The encryption key is a 32-byte random value held in ~/.ollamacode/keyfile
(mode 0o600). It is generated on first use and never leaves the machine.

Usage:
    from ollamacode.secrets import set_secret, get_secret, list_secrets, delete_secret

Config resolution:
    Use ``resolve_secret(value)`` to transparently unwrap "secret:<name>"
    references in config values (e.g. api_key: "secret:groq_key").
"""

from __future__ import annotations

import base64
import os
import stat
from pathlib import Path

_OLLAMACODE_DIR = Path(os.path.expanduser("~")) / ".ollamacode"
_KEYFILE_PATH = _OLLAMACODE_DIR / "keyfile"
_KEY_BYTES = 32  # 256-bit AES key


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------


def _ensure_keyfile() -> bytes:
    """Load the machine-local keyfile, creating it on first use."""
    _OLLAMACODE_DIR.mkdir(parents=True, exist_ok=True)

    if _KEYFILE_PATH.exists():
        mode = stat.S_IMODE(_KEYFILE_PATH.stat().st_mode)
        if mode != 0o600:
            import warnings

            warnings.warn(
                f"Keyfile {_KEYFILE_PATH} has permissions {oct(mode)}, expected 0o600. "
                "Fix with: chmod 600 ~/.ollamacode/keyfile",
                UserWarning,
                stacklevel=3,
            )
        return _KEYFILE_PATH.read_bytes()

    # First run: generate a fresh random key.
    key = os.urandom(_KEY_BYTES)
    _KEYFILE_PATH.write_bytes(key)
    try:
        _KEYFILE_PATH.chmod(0o600)
    except OSError:
        pass
    return key


def _get_cipher_key() -> bytes:
    """Return a 32-byte AES key derived from the keyfile."""
    raw = _ensure_keyfile()
    if len(raw) >= _KEY_BYTES:
        return raw[:_KEY_BYTES]
    # Keyfile shorter than expected (should not happen) — hash it to 32 bytes.
    import hashlib

    return hashlib.sha256(raw).digest()


# ---------------------------------------------------------------------------
# Low-level encrypt / decrypt
# ---------------------------------------------------------------------------


def encrypt(plaintext: str) -> str:
    """Encrypt *plaintext* with AES-256-GCM.

    Returns a ``v1:<base64>`` string that encodes ``nonce (12 B) + ciphertext + tag (16 B)``.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise RuntimeError(
            "The 'cryptography' package is required for secrets storage. "
            "Install with: pip install cryptography"
        ) from exc

    key = _get_cipher_key()
    nonce = os.urandom(12)  # 96-bit nonce recommended for GCM
    aesgcm = AESGCM(key)
    ct_with_tag = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    encoded = base64.b64encode(nonce + ct_with_tag).decode("ascii")
    return f"v1:{encoded}"


def decrypt(ciphertext: str) -> str:
    """Decrypt a string produced by :func:`encrypt`.

    Raises :class:`ValueError` on authentication failure or bad format.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise RuntimeError(
            "The 'cryptography' package is required for secrets storage. "
            "Install with: pip install cryptography"
        ) from exc

    if not ciphertext.startswith("v1:"):
        raise ValueError(
            f"Unrecognized ciphertext format (expected 'v1:...'), got: {ciphertext[:20]!r}"
        )

    raw = base64.b64decode(ciphertext[3:])
    if len(raw) < 12 + 16:
        raise ValueError("Ciphertext too short to be valid")

    nonce = raw[:12]
    ct_with_tag = raw[12:]
    key = _get_cipher_key()
    aesgcm = AESGCM(key)
    try:
        plaintext_bytes = aesgcm.decrypt(nonce, ct_with_tag, None)
    except Exception as exc:
        raise ValueError(
            f"Decryption failed (wrong key or tampered data): {exc}"
        ) from exc
    return plaintext_bytes.decode("utf-8")


# ---------------------------------------------------------------------------
# High-level secrets store (backed by state.json encrypted_secrets key)
# ---------------------------------------------------------------------------


def set_secret(name: str, value: str) -> None:
    """Store an encrypted secret under *name*."""
    from .state import _load_raw, _save

    name = name.strip()
    if not name:
        raise ValueError("Secret name cannot be empty")

    data = _load_raw()
    secrets: dict = data.get("encrypted_secrets") or {}
    secrets[name] = encrypt(value)
    data["encrypted_secrets"] = secrets
    _save(data)


def get_secret(name: str) -> str | None:
    """Retrieve and decrypt the secret stored under *name*.

    Returns ``None`` if the secret does not exist.
    """
    from .state import _load_raw

    name = name.strip()
    data = _load_raw()
    secrets: dict = data.get("encrypted_secrets") or {}
    ct = secrets.get(name)
    if ct is None:
        return None
    return decrypt(ct)


def delete_secret(name: str) -> bool:
    """Delete the secret stored under *name*.

    Returns ``True`` if the secret existed and was removed, ``False`` if it was not found.
    """
    from .state import _load_raw, _save

    name = name.strip()
    data = _load_raw()
    secrets: dict = data.get("encrypted_secrets") or {}
    if name not in secrets:
        return False
    del secrets[name]
    data["encrypted_secrets"] = secrets
    _save(data)
    return True


def list_secrets() -> list[str]:
    """Return sorted names of all stored secrets (values are not revealed)."""
    from .state import _load_raw

    data = _load_raw()
    secrets: dict = data.get("encrypted_secrets") or {}
    return sorted(secrets.keys())


# ---------------------------------------------------------------------------
# Config resolution helper
# ---------------------------------------------------------------------------


def resolve_secret(value: str | None) -> str | None:
    """Transparently unwrap a ``"secret:<name>"`` reference.

    If *value* starts with ``"secret:"``, the remainder is used as the secret
    name and its decrypted value is returned. Otherwise *value* is returned
    unchanged. Raises :class:`KeyError` if the referenced secret does not exist.

    Example in ``ollamacode.yaml``::

        api_key: "secret:groq_key"
    """
    if value and isinstance(value, str) and value.startswith("secret:"):
        name = value[7:]
        resolved = get_secret(name)
        if resolved is None:
            raise KeyError(
                f"Secret '{name}' not found. Set it with: ollamacode secrets set {name}"
            )
        return resolved
    return value
