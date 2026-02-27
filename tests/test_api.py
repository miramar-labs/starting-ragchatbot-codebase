"""
Tests for FastAPI endpoints (app.py).

Import-time setup
-----------------
app.py has two things that would fail in a normal test environment:
  1. StaticFiles(directory="../frontend") – the directory won't exist relative to
     wherever pytest is run from.
  2. RAGSystem(config) – opens ChromaDB and loads the sentence-transformer model.

Both are patched at *module collection time* (before app.py is first imported)
so they never attempt real I/O.  The patches are applied directly to the
already-imported module objects, which means they persist for the lifetime of
the test session without interfering with other test files.
"""

import sys
import os
from unittest.mock import MagicMock, patch

# ------------------------------------------------------------------
# 1. Ensure backend/ is on sys.path (conftest.py does this too, but
#    be explicit since module-level code runs very early).
# ------------------------------------------------------------------
_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if _backend not in sys.path:
    sys.path.insert(0, _backend)

# ------------------------------------------------------------------
# 2. Stub out StaticFiles so it doesn't check for directory existence.
# ------------------------------------------------------------------
import fastapi.staticfiles as _static_mod

_RealStaticFiles = _static_mod.StaticFiles
_static_mod.StaticFiles = MagicMock(name="StaticFiles")

# ------------------------------------------------------------------
# 3. Import app *inside* a RAGSystem patch so the module-level
#    `rag_system = RAGSystem(config)` line returns a mock.
# ------------------------------------------------------------------
_mock_rag = MagicMock(name="rag_system")

with patch("rag_system.RAGSystem", return_value=_mock_rag):
    import app as _app_module
    from app import app

# Restore StaticFiles (only matters if other test code uses it directly)
_static_mod.StaticFiles = _RealStaticFiles

# Point the module-level variable at our mock (already the case, but explicit)
_app_module.rag_system = _mock_rag

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Return a TestClient with the mock RAGSystem fully reset before each test.

    reset_mock(side_effect=True, return_value=True) propagates to child mocks,
    preventing side_effect leakage between tests (e.g. a RuntimeError set in an
    exception-handling test bleeding into the next test).
    """
    _mock_rag.reset_mock(side_effect=True, return_value=True)
    _app_module.rag_system = _mock_rag
    return TestClient(app, raise_server_exceptions=True)


# ------------------------------------------------------------------
# GET /api/courses
# ------------------------------------------------------------------

class TestCoursesEndpoint:
    def test_returns_200(self, client):
        _mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        assert client.get("/api/courses").status_code == 200

    def test_total_courses_field(self, client):
        _mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["A", "B", "C"],
        }
        data = client.get("/api/courses").json()
        assert data["total_courses"] == 3

    def test_course_titles_field(self, client):
        _mock_rag.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Python 101"],
        }
        data = client.get("/api/courses").json()
        assert "Python 101" in data["course_titles"]

    def test_empty_catalog(self, client):
        _mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        data = client.get("/api/courses").json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_exception_returns_500(self, client):
        _mock_rag.get_course_analytics.side_effect = RuntimeError("DB exploded")
        assert client.get("/api/courses").status_code == 500

    def test_calls_get_course_analytics(self, client):
        _mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        client.get("/api/courses")
        _mock_rag.get_course_analytics.assert_called_once()


# ------------------------------------------------------------------
# POST /api/query
# ------------------------------------------------------------------

class TestQueryEndpoint:
    def _setup(self, session_id="session_1", answer="An answer.", sources=None):
        _mock_rag.session_manager.create_session.return_value = session_id
        _mock_rag.query.return_value = (answer, sources or [])

    def test_returns_200(self, client):
        self._setup()
        r = client.post("/api/query", json={"query": "What is Python?"})
        assert r.status_code == 200

    def test_answer_in_response(self, client):
        self._setup(answer="Python is awesome.")
        data = client.post("/api/query", json={"query": "What is Python?"}).json()
        assert data["answer"] == "Python is awesome."

    def test_sources_in_response(self, client):
        self._setup(sources=["Python 101 - Lesson 1"])
        data = client.post("/api/query", json={"query": "test"}).json()
        assert "Python 101 - Lesson 1" in data["sources"]

    def test_session_id_in_response(self, client):
        self._setup(session_id="session_42")
        data = client.post("/api/query", json={"query": "test"}).json()
        assert data["session_id"] == "session_42"

    def test_new_session_created_when_not_provided(self, client):
        self._setup()
        client.post("/api/query", json={"query": "test"})
        _mock_rag.session_manager.create_session.assert_called_once()

    def test_provided_session_id_is_reused(self, client):
        _mock_rag.query.return_value = ("Answer.", [])
        data = client.post(
            "/api/query", json={"query": "test", "session_id": "existing_session"}
        ).json()
        assert data["session_id"] == "existing_session"
        _mock_rag.session_manager.create_session.assert_not_called()

    def test_rag_query_called_with_query_text(self, client):
        self._setup()
        client.post("/api/query", json={"query": "Hello world"})
        args, _ = _mock_rag.query.call_args
        assert "Hello world" in args[0]

    def test_empty_sources_list(self, client):
        self._setup(sources=[])
        data = client.post("/api/query", json={"query": "test"}).json()
        assert data["sources"] == []

    def test_exception_returns_500(self, client):
        _mock_rag.session_manager.create_session.return_value = "s1"
        _mock_rag.query.side_effect = RuntimeError("AI failure")
        assert client.post("/api/query", json={"query": "test"}).status_code == 500

    def test_missing_query_field_returns_422(self, client):
        r = client.post("/api/query", json={})
        assert r.status_code == 422

    def test_multiple_sources_returned(self, client):
        self._setup(sources=["Course A - Lesson 1", "Course B - Lesson 3"])
        data = client.post("/api/query", json={"query": "test"}).json()
        assert len(data["sources"]) == 2
