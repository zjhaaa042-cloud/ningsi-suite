"""Application services shared by the engine's CLI and product adapters."""

from .window import ProcessedWindow, process_window

__all__ = ["ProcessedWindow", "process_window"]
