"""Tests for tiny_memoize.core.memoize.

We deliberately test the behaviour the implementation claims to
support: custom key functions, bounded LRU eviction with touch-on-read,
non-caching of exceptions, and metadata preservation.  We do not test
behaviours the library does not promise (e.g. keyword-order folding,
shared caches, thread safety).
"""

import unittest

from tiny_memoize import memoize, MemoizedError


class TestMemoize(unittest.TestCase):
    def test_caches_by_key_fn_result(self):
        calls = []

        @memoize(lambda fn, a, kw: a[0] * 100, maxsize=10)
        def f(x):
            calls.append(x)
            return x * 2

        self.assertEqual(f(3), 6)
        self.assertEqual(f(3), 6)
        self.assertEqual(calls, [3])

    def test_different_args_different_cache_entries(self):
        calls = []

        @memoize(lambda fn, a, kw: a, maxsize=10)
        def f(x):
            calls.append(x)
            return x * 2

        self.assertEqual(f(2), 4)
        self.assertEqual(f(3), 6)
        self.assertEqual(calls, [2, 3])

    def test_key_fn_receives_args_and_kwargs(self):
        seen = []

        def key_fn(fn, args, kwargs):
            seen.append((args, kwargs))
            return (args, tuple(sorted(kwargs.items())))

        @memoize(key_fn, maxsize=10)
        def f(x, y=1):
            return x + y

        self.assertEqual(f(1, y=2), 3)
        self.assertEqual(
            seen, [((1,), {"y": 2})],
        )

    def test_key_fn_receives_original_callable(self):
        captured = {}

        def key_fn(fn, args, kwargs):
            captured["fn"] = fn
            return args

        @memoize(key_fn, maxsize=10)
        def f(x):
            return x

        f(1)
        self.assertIs(captured["fn"], f.__wrapped__)

    def test_lru_eviction_when_store_exceeds_maxsize(self):
        calls = []

        @memoize(lambda fn, a, kw: a[0], maxsize=3)
        def f(x):
            calls.append(x)
            return x

        self.assertEqual(f(1), 1)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(3), 3)
        # Inserting 4 evicts the LRU (1), since none have been re-read.
        self.assertEqual(f(4), 4)
        self.assertEqual(f(1), 1)  # Recomputed because 1 was evicted.
        self.assertEqual(calls, [1, 2, 3, 4, 1])

    def test_touch_on_read_prevents_eviction_of_recently_used(self):
        calls = []

        @memoize(lambda fn, a, kw: a[0], maxsize=3)
        def f(x):
            calls.append(x)
            return x

        f(1)
        f(2)
        f(3)
        f(1)  # Read of 1 refreshes its recency; now 2 is LRU.
        f(4)  # Inserting 4 evicts 2, not 1.
        self.assertEqual(f(2), 2)  # Recomputed because 2 was evicted.
        self.assertEqual(f(1), 1)  # Still cached.
        self.assertEqual(calls, [1, 2, 3, 4, 2])

    def test_exceptions_are_not_cached(self):
        calls = []

        class Boom(Exception):
            pass

        @memoize(lambda fn, a, kw: a[0], maxsize=5)
        def f(x):
            calls.append(x)
            if x == 7:
                raise Boom("no")
            return x

        with self.assertRaises(Boom):
            f(7)
        with self.assertRaises(Boom):
            f(7)
        self.assertEqual(calls, [7, 7])

    def test_cache_attribute_exposed_and_populated(self):
        @memoize(lambda fn, a, kw: a[0], maxsize=5)
        def f(x):
            return x * 10

        self.assertEqual(f.cache, {})
        f(2)
        self.assertEqual(f.cache, {2: 20})

    def test_wrapped_attribute_is_original_function(self):
        def f(x):
            return x

        g = memoize(lambda fn, a, kw: a[0], maxsize=5)(f)
        self.assertIs(g.__wrapped__, f)

    def test_maxsize_must_be_positive_integer(self):
        with self.assertRaises(MemoizedError):
            memoize(lambda fn, a, kw: a, maxsize=0)
        with self.assertRaises(MemoizedError):
            memoize(lambda fn, a, kw: a, maxsize=-1)

    def test_maxsize_rejects_bool(self):
        # ``bool`` is a subclass of ``int``; we explicitly reject it so
        # that passing ``True`` does not silently become maxsize=1.
        with self.assertRaises(MemoizedError):
            memoize(lambda fn, a, kw: a, maxsize=True)

    def test_maxsize_rejects_non_integer(self):
        with self.assertRaises(MemoizedError):
            memoize(lambda fn, a, kw: a, maxsize=1.5)

    def test_key_fn_must_be_callable(self):
        with self.assertRaises(MemoizedError):
            memoize("not callable", maxsize=5)  # type: ignore[arg-type]

    def test_non_hashable_key_raises_typeerror(self):
        @memoize(lambda fn, a, kw: [], maxsize=5)  # returns a list
        def f(x):
            return x

        with self.assertRaises(TypeError):
            f(1)

    def test_preserves_function_metadata(self):
        @memoize(lambda fn, a, kw: a[0], maxsize=5)
        def add(a, b):
            """Sum two numbers."""
            return a + b

        self.assertEqual(add.__name__, "add")
        self.assertEqual(add.__doc__, "Sum two numbers.")

    def test_eviction_order_after_only_reads(self):
        # After populating a full cache, repeated reads should keep all
        # entries alive without recomputation.
        calls = []

        @memoize(lambda fn, a, kw: a[0], maxsize=3)
        def f(x):
            calls.append(x)
            return x

        f(1)
        f(2)
        f(3)
        f(1)
        f(2)
        f(3)
        self.assertEqual(calls, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
