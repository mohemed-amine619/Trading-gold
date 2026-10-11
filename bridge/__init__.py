"""
MT5 Bridge — Windows-side service exposing MetaTrader 5 to the Linux core.

This package is the ONLY place in the project permitted to import
MetaTrader5, and it runs only inside the Windows 11 VM.

See docs/MT5_WINDOWS_BRIDGE.md for the full design and setup.
"""

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Lazy re-export so importing the package never builds an adapter."""
    from bridge.app import create_app as _create_app
    return _create_app(*args, **kwargs)