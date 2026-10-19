"""tiny_memoize: single-key memoisation with a bounded store."""

from .core import memoize, MemoizedError

__all__ = ["memoize", "MemoizedError"]
