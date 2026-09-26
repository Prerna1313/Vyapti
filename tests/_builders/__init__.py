"""Test-only builders for synthetic data.

This package exists so that synthetic data construction is NEVER
reachable from the real-data production code path. Anything that
needs in-memory synthetic data (PDW streams, in-memory H5 files)
should import from here.
"""
