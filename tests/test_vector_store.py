"""Unit tests for SearchResults and VectorStore helpers (vector_store.py).

VectorStore.__init__ opens a real ChromaDB connection, so we never instantiate
it directly.  _build_filter is pure logic and can be called by bypassing __init__
with __new__.  SearchResults is a plain dataclass with no external dependencies.
"""
from vector_store import SearchResults, VectorStore


class TestSearchResults:
    def test_is_empty_true_when_no_documents(self):
        sr = SearchResults(documents=[], metadata=[], distances=[])
        assert sr.is_empty() is True

    def test_is_empty_false_when_has_documents(self):
        sr = SearchResults(documents=["some text"], metadata=[{}], distances=[0.1])
        assert sr.is_empty() is False

    def test_empty_factory_sets_error(self):
        sr = SearchResults.empty("Something went wrong")
        assert sr.error == "Something went wrong"
        assert sr.is_empty() is True


class TestBuildFilter:
    """Tests for VectorStore._build_filter.

    _build_filter uses no instance state, so we bypass __init__ with __new__.
    """

    def setup_method(self):
        self.vs = VectorStore.__new__(VectorStore)

    def test_no_args_returns_none(self):
        assert self.vs._build_filter(None, None) is None

    def test_course_only_returns_course_filter(self):
        result = self.vs._build_filter("Python 101", None)
        assert result == {"course_title": "Python 101"}

    def test_lesson_only_returns_lesson_filter(self):
        result = self.vs._build_filter(None, 3)
        assert result == {"lesson_number": 3}

    def test_both_returns_and_clause(self):
        result = self.vs._build_filter("Python 101", 2)
        assert result == {"$and": [
            {"course_title": "Python 101"},
            {"lesson_number": 2},
        ]}
