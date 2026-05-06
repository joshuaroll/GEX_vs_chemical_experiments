"""Pytest configuration for dili_downstream.

Registers custom markers used across the test suite. Adding the marker
declaration silences pytest's `PytestUnknownMarkWarning`.
"""

from __future__ import annotations


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "network: mark a test that requires internet access (skipped by "
        "default; set TDC_NETWORK_TESTS=1 or other gating env to enable).",
    )
