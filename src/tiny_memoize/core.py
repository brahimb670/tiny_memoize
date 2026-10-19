"""Core implementation of a memoising decorator with a bounded store.

Design choices made deliberately, not by compromise:

1. **Custom key function only.**
   The caller supplies a ``key_fn(call, args, kwargs)`` whose job is to
   hash the arguments.  The store key is whatever ``key_fn`` returns; we
   feed it straight into a ``dict``.  If a callable needs to ignore
   arguments, fold keyword order, or canonicalise types, it does that in
   its own ``key_fn``.  Keeping this out of the library means we never
   have to guess whether ``f(x=1, y=2)`` and ``f(y=2, x=1)`` are the same
   call; the author of the wrapped function decides.

2. **Per-callable state.**
   ``memoize`` returns a fresh wrapper per decorated callable, each with
   its own store.  We deliberately do not share a cache across functions.
   Sharing would make eviction surprising (``f`` evicts ``g``) and make
   reasoning about cache size hard.  If you want to share, compose your
   own structure around several memoised callables.

3. **Exceptions are NOT cached.**
   A transient ``KeyError`` or network failure should not poison the
   cache for ``maxsize`` subsequent calls.  Re-raising every time is the
   least surprising behaviour for a small, general-purpose memoiser.

4. **LRU eviction with touch-on-read.**
   A read is a use.  This is the LRU contract most people expect; it
   differs from a pure insertion-order FIFO and from LRU-without-reads.
   Implementation: on every cache hit we delete and re-insert the key so
   it moves to the end of CPython's ordered dict.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Hashable, Optional, Tuple


class MemoizedError(Exception):
    """Base class for configuration errors raised by :func:`memoize`.

    Raising a dedicated error type keeps failures of the decorator
    itself (bad arguments to it) separable from exceptions raised *by*
    the wrapped function.  We never raise this from inside a wrapped
    call.
    """


def memoize(
    key_fn: Callable[[Callable[..., Any], tuple, Dict[str, Any]], Hashable],
    maxsize: int = 128,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that memoises a callable.

    Parameters
    ----------
    key_fn:
        A function ``(call, args, kwargs) -> hashable``.  ``call`` is the
        original undecorated function, included so the key can depend on
        function identity if the same ``key_fn`` is shared across several
        callables.  The return value must be hashable; we use it as a
        ``dict`` key.
    maxsize:
        Maximum number of stored results.  Must be a positive integer.
        When the store is full and a *new* key arrives, the
        least-recently-used entry is evicted.  Reads (hits) refresh the
        recency of the read entry.

    Returns
    -------
    A decorator that preserves the wrapped function's metadata via
    :func:`functools.wraps` and exposes the underlying store as the
    ``cache`` attribute.

    Raises
    ------
    MemoizedError
        If ``maxsize`` is not a positive integer.
    """
    if not isinstance(maxsize, int) or isinstance(maxsize, bool) or maxsize <= 0:
        raise MemoizedError("maxsize must be a positive integer")
    if not callable(key_fn):
        raise MemoizedError("key_fn must be callable")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # Per-callable store; ``dict`` is insertion-ordered in CPython 3.7+
        # and the standard library guarantees order from 3.7 onward.
        cache: Dict[Hashable, Any] = {}

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                k = key_fn(fn, args, kwargs)
                hash(k)
            except TypeError as exc:
                # A non-hashable key from a buggy ``key_fn`` should fail
                # loudly at call time rather than silently bypass the
                # cache.  Re-raise the TypeError with context.
                raise TypeError(
                    f"key_fn returned a non-hashable value: {exc}"
                ) from exc

            if k in cache:
                # Touch-on-read LRU: delete and re-insert so recently
                # used entries survive eviction.
                v = cache.pop(k)
                cache[k] = v
                return v

            result = fn(*args, **kwargs)
            # We do NOT wrap this in try/except: exceptions propagate to
            # the caller and nothing is stored.  See module docstring #3.
            cache[k] = result
            if len(cache) > maxsize:
                # The first key inserted is the least-recently-used at
                # this point because we always re-insert on reads.
                first = next(iter(cache))
                del cache[first]
            return result

        # Expose the store for inspection and clearing in tests and user
        # code.  Not for mutation; we trust the caller with the sharp
        # edge.
        wrapper.cache = cache  # type: ignore[attr-defined]
        # Keep the original callable reachable for key_fn-derived logic
        # and introspection.
        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapper

    return decorator
