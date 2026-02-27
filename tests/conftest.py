"""Pytest configuration: add backend/ to sys.path for all tests."""
import sys
import os
from unittest.mock import MagicMock

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# chromadb is a heavy C-extension that may not be installed in every
# environment (e.g. CI or a plain pip install without the full deps).
# Mock it so backend modules can be imported without the real package.
for _mod in (
    "chromadb",
    "chromadb.config",
    "chromadb.utils",
    "chromadb.utils.embedding_functions",
    "sentence_transformers",
    "anthropic",
):
    sys.modules.setdefault(_mod, MagicMock())
