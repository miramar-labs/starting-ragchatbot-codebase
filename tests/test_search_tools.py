"""Unit tests for ToolManager and CourseSearchTool (search_tools.py)."""
import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool, Tool, ToolManager
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _SimpleTool(Tool):
    """Minimal concrete Tool for ToolManager tests."""

    def get_tool_definition(self):
        return {"name": "simple_tool", "description": "A simple tool"}

    def execute(self, **kwargs):
        return "simple_result"


class _ToolMissingName(Tool):
    def get_tool_definition(self):
        return {"description": "No name here"}

    def execute(self, **kwargs):
        return "x"


def _make_results(docs=None, meta=None, error=None):
    docs = docs or []
    meta = meta or []
    return SearchResults(
        documents=docs,
        metadata=meta,
        distances=[0.1] * len(docs),
        error=error,
    )


# ---------------------------------------------------------------------------
# ToolManager
# ---------------------------------------------------------------------------

class TestToolManager:
    def test_register_tool_stores_it(self):
        tm = ToolManager()
        tm.register_tool(_SimpleTool())
        assert "simple_tool" in tm.tools

    def test_register_tool_without_name_raises_value_error(self):
        tm = ToolManager()
        with pytest.raises(ValueError):
            tm.register_tool(_ToolMissingName())

    def test_get_tool_definitions_returns_list(self):
        tm = ToolManager()
        tm.register_tool(_SimpleTool())
        defs = tm.get_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) == 1

    def test_get_tool_definitions_empty_by_default(self):
        assert ToolManager().get_tool_definitions() == []

    def test_execute_known_tool(self):
        tm = ToolManager()
        tm.register_tool(_SimpleTool())
        assert tm.execute_tool("simple_tool") == "simple_result"

    def test_execute_unknown_tool_returns_error_string(self):
        tm = ToolManager()
        result = tm.execute_tool("ghost")
        assert "ghost" in result or "not found" in result.lower()

    def test_get_last_sources_empty_initially(self):
        assert ToolManager().get_last_sources() == []

    def test_reset_sources_clears_tool_sources(self):
        tm = ToolManager()
        store = MagicMock()
        store.search.return_value = _make_results(
            docs=["text"],
            meta=[{"course_title": "Course A", "lesson_number": 1}],
        )
        tool = CourseSearchTool(store)
        tm.register_tool(tool)
        tool.execute(query="anything")
        assert len(tm.get_last_sources()) == 1
        tm.reset_sources()
        assert tm.get_last_sources() == []

    def test_multiple_tools_registered(self):
        class _AnotherTool(Tool):
            def get_tool_definition(self):
                return {"name": "another"}
            def execute(self, **kwargs):
                return "y"

        tm = ToolManager()
        tm.register_tool(_SimpleTool())
        tm.register_tool(_AnotherTool())
        assert len(tm.get_tool_definitions()) == 2

    def test_execute_passes_kwargs_to_tool(self):
        class _EchoTool(Tool):
            def get_tool_definition(self):
                return {"name": "echo"}
            def execute(self, value=None, **kwargs):
                return value

        tm = ToolManager()
        tm.register_tool(_EchoTool())
        assert tm.execute_tool("echo", value="hello") == "hello"


# ---------------------------------------------------------------------------
# CourseSearchTool
# ---------------------------------------------------------------------------

class TestCourseSearchTool:
    @pytest.fixture
    def store(self):
        return MagicMock()

    @pytest.fixture
    def tool(self, store):
        return CourseSearchTool(store)

    # --- schema ---

    def test_tool_name_is_search_course_content(self, tool):
        assert tool.get_tool_definition()["name"] == "search_course_content"

    def test_query_is_in_required_fields(self, tool):
        schema = tool.get_tool_definition()["input_schema"]
        assert "query" in schema["required"]

    def test_schema_has_optional_course_name(self, tool):
        props = tool.get_tool_definition()["input_schema"]["properties"]
        assert "course_name" in props

    def test_schema_has_optional_lesson_number(self, tool):
        props = tool.get_tool_definition()["input_schema"]["properties"]
        assert "lesson_number" in props

    # --- execute: error / empty cases ---

    def test_execute_returns_error_message_on_search_error(self, tool, store):
        store.search.return_value = _make_results(error="Search error: DB down")
        result = tool.execute(query="anything")
        assert "Search error" in result

    def test_execute_returns_no_content_message_when_empty(self, tool, store):
        store.search.return_value = _make_results()
        result = tool.execute(query="obscure topic")
        assert "No relevant content found" in result

    def test_empty_result_message_mentions_course_filter(self, tool, store):
        store.search.return_value = _make_results()
        result = tool.execute(query="test", course_name="Unknown Course")
        assert "Unknown Course" in result

    def test_empty_result_message_mentions_lesson_filter(self, tool, store):
        store.search.return_value = _make_results()
        result = tool.execute(query="test", lesson_number=7)
        assert "7" in result

    # --- execute: results formatting ---

    def test_result_includes_course_title(self, tool, store):
        store.search.return_value = _make_results(
            docs=["Python basics"],
            meta=[{"course_title": "Python 101", "lesson_number": 1}],
        )
        result = tool.execute(query="python")
        assert "Python 101" in result

    def test_result_includes_lesson_number(self, tool, store):
        store.search.return_value = _make_results(
            docs=["Content"],
            meta=[{"course_title": "Course A", "lesson_number": 3}],
        )
        result = tool.execute(query="test")
        assert "Lesson 3" in result

    def test_result_includes_document_text(self, tool, store):
        store.search.return_value = _make_results(
            docs=["unique_content_string"],
            meta=[{"course_title": "C", "lesson_number": 1}],
        )
        result = tool.execute(query="test")
        assert "unique_content_string" in result

    # --- source tracking ---

    def test_last_sources_populated_after_execute(self, tool, store):
        store.search.return_value = _make_results(
            docs=["content"],
            meta=[{"course_title": "Course A", "lesson_number": 2}],
        )
        tool.execute(query="test")
        assert len(tool.last_sources) == 1
        assert "Course A" in tool.last_sources[0]

    def test_last_sources_empty_on_no_results(self, tool, store):
        store.search.return_value = _make_results()
        tool.execute(query="test")
        # Sources stay empty when there are no results
        assert tool.last_sources == []

    def test_last_sources_includes_lesson_number(self, tool, store):
        store.search.return_value = _make_results(
            docs=["text"],
            meta=[{"course_title": "ML Basics", "lesson_number": 5}],
        )
        tool.execute(query="test")
        assert "Lesson 5" in tool.last_sources[0]

    # --- filter forwarding ---

    def test_passes_course_name_filter_to_store(self, tool, store):
        store.search.return_value = _make_results()
        tool.execute(query="test", course_name="Python 101")
        store.search.assert_called_once_with(
            query="test", course_name="Python 101", lesson_number=None
        )

    def test_passes_lesson_number_filter_to_store(self, tool, store):
        store.search.return_value = _make_results()
        tool.execute(query="test", lesson_number=4)
        store.search.assert_called_once_with(
            query="test", course_name=None, lesson_number=4
        )

    def test_passes_both_filters_to_store(self, tool, store):
        store.search.return_value = _make_results()
        tool.execute(query="test", course_name="Course", lesson_number=2)
        store.search.assert_called_once_with(
            query="test", course_name="Course", lesson_number=2
        )
