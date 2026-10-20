# tiny_memoize

A single-key memoising decorator with a custom key function and a bounded, LRU-style store. Standard library only; no third-party dependencies.

## Usage

```python
from tiny_memoize import memoize

def key_fn(fn, args, kwargs):
    # fn is the original undecorated callable.
    # Return any hashable; this is the cache key.
    return args

@memoize(key_fn, maxsize=128)
def expensive(x, y=1):
    return x * y

expensive(2, y=3)   # computed
expensive(2, y=3)   # served from cache
expensive.cache     # dict: the live store, for inspection/clearing
expensive.__wrapped__  # the original undecorated function
```

## Why this exists

Most memoisers (`functools.lru_cache` included) hash arguments for you. That is convenient until you want to fold keyword order, ignore selected arguments, or canonicalise types — at which point you fight the built-in keying logic. `tiny_memoize` takes the opposite position: you supply a `key_fn(call, args, kwargs)` and the library uses exactly what it returns as the store key. The cost is one extra function per call; the benefit is that cache identity is never a mystery.

The store is bounded and evicts least-recently-used entries, where a read counts as a use. Exceptions are never cached: a transient failure should not poison the cache for the next `maxsize` callers.

## The awkward edge

The `key_fn` is on its own with respect to argument equivalence. `f(x=1, y=2)` and `f(y=2, x=1)` will only share a cache entry if your `key_fn` makes them equal. The library will not try to guess. If your `key_fn` returns a non-hashable value, the wrapped call raises `TypeError` rather than silently bypassing the cache.

## Exports

- `memoize(key_fn, maxsize=128)` — the decorator factory.
- `MemoizedError` — raised for configuration errors (bad `maxsize`, non-callable `key_fn`).

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```
