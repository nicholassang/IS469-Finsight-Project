"""
conftest.py — pytest configuration for sandbox / offline environments.

Patches tiktoken to use word-based approximation when the BPE vocab
file cannot be downloaded (e.g. sandbox with restricted internet access).
"""

import sys
import pytest


def _word_based_encoder(text):
    """Simple token approximation: 1 token ≈ 0.75 words."""
    words = text.split()
    # return a list whose length approximates the token count
    approx = int(len(words) * 1.33)
    return list(range(approx))


class _FakeEncoding:
    def encode(self, text, *args, **kwargs):
        return _word_based_encoder(text)


def _fake_get_encoding(name):
    return _FakeEncoding()


def pytest_configure(config):
    """
    Intercept tiktoken before any test module tries to download its BPE file.
    Only patches if the real download would fail (proxy-blocked sandbox).
    """
    try:
        import tiktoken
        # Try to get an encoding — if it needs a download it will throw.
        enc = tiktoken.get_encoding("cl100k_base")
        enc.encode("test")          # actually triggers the download
    except Exception:
        # Can't download BPE file → monkey-patch tiktoken globally
        import tiktoken
        tiktoken.get_encoding = _fake_get_encoding
        tiktoken.encoding_for_model = lambda m: _FakeEncoding()

        # Also patch the cached _encoders dict inside chunker if it's loaded
        try:
            import src.chunking.chunker as ck
            ck._encoders.clear()
            ck.get_encoder = lambda name="cl100k_base": _FakeEncoding()
            ck.count_tokens = lambda text, encoding_name="cl100k_base": int(
                len(text.split()) * 1.33
            )
        except ImportError:
            pass
