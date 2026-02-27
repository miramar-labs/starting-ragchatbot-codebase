"""Unit tests for DocumentProcessor (document_processor.py)."""
import pytest
from document_processor import DocumentProcessor


SMALL_CHUNK = 60   # small enough to force multi-chunk splits
OVERLAP = 10


@pytest.fixture
def processor():
    """Standard processor matching config defaults."""
    return DocumentProcessor(chunk_size=800, chunk_overlap=100)


@pytest.fixture
def small_processor():
    """Processor with tiny chunk size to easily force splits."""
    return DocumentProcessor(chunk_size=SMALL_CHUNK, chunk_overlap=OVERLAP)


# ---------------------------------------------------------------------------
# chunk_text()
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_string_returns_empty_list(self, processor):
        assert processor.chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self, processor):
        assert processor.chunk_text("   \n\t  ") == []

    def test_short_text_returns_single_chunk(self, processor):
        text = "This is a short sentence."
        chunks = processor.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self, small_processor):
        # Each sentence is ~25 chars; 4 sentences force at least 2 chunks at size 60
        text = (
            "First sentence here now. "
            "Second sentence follows it. "
            "Third sentence comes next. "
            "Fourth sentence ends here."
        )
        chunks = small_processor.chunk_text(text)
        assert len(chunks) > 1

    def test_all_content_is_represented(self, small_processor):
        text = (
            "Alpha sentence is present. "
            "Beta sentence is present. "
            "Gamma sentence is present. "
            "Delta sentence is present."
        )
        chunks = small_processor.chunk_text(text)
        combined = " ".join(chunks)
        assert "Alpha" in combined
        assert "Delta" in combined

    def test_normalizes_multiple_spaces(self, processor):
        text = "Word   with   extra    spaces."
        chunks = processor.chunk_text(text)
        assert len(chunks) == 1
        assert "   " not in chunks[0]

    def test_normalizes_newlines(self, processor):
        text = "First line.\nSecond line."
        chunks = processor.chunk_text(text)
        # Newlines are collapsed into spaces; result is a single normalized chunk
        assert len(chunks) >= 1
        assert "\n" not in chunks[0]

    def test_chunk_list_contains_strings(self, processor):
        chunks = processor.chunk_text("Hello world. This is a test.")
        assert all(isinstance(c, str) for c in chunks)

    def test_single_very_long_sentence_still_included(self, small_processor):
        """A sentence longer than chunk_size must not be silently dropped."""
        long_sentence = "W" * 200  # well over SMALL_CHUNK
        chunks = small_processor.chunk_text(long_sentence)
        assert len(chunks) >= 1
        combined = "".join(chunks)
        assert "W" * 10 in combined

    def test_no_overlap_processor_still_produces_chunks(self):
        proc = DocumentProcessor(chunk_size=50, chunk_overlap=0)
        text = "One sentence. Two sentence. Three sentence. Four sentence."
        chunks = proc.chunk_text(text)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# process_course_document()
# ---------------------------------------------------------------------------

VALID_DOC = """\
Course Title: Python Fundamentals
Course Link: https://example.com/python
Course Instructor: Jane Smith

Lesson 1: Getting Started
Lesson Link: https://example.com/python/1
Python is a high-level programming language. It was created by Guido van Rossum.
Python emphasizes readability and simplicity. You will love it.

Lesson 2: Variables
Lesson Link: https://example.com/python/2
Variables store data values in Python. They are created with assignment statements.
You can assign integers, strings, and other types.
"""


class TestProcessCourseDocumentMetadata:
    def test_extracts_course_title(self, processor, tmp_path):
        doc = tmp_path / "course.txt"
        doc.write_text(VALID_DOC)
        course, _ = processor.process_course_document(str(doc))
        assert course.title == "Python Fundamentals"

    def test_extracts_course_link(self, processor, tmp_path):
        doc = tmp_path / "course.txt"
        doc.write_text(VALID_DOC)
        course, _ = processor.process_course_document(str(doc))
        assert course.course_link == "https://example.com/python"

    def test_extracts_instructor(self, processor, tmp_path):
        doc = tmp_path / "course.txt"
        doc.write_text(VALID_DOC)
        course, _ = processor.process_course_document(str(doc))
        assert course.instructor == "Jane Smith"

    def test_missing_instructor_defaults_to_none(self, processor, tmp_path):
        content = (
            "Course Title: No Instructor\n"
            "Course Link: https://example.com\n"
            "\n"
            "Lesson 1: Intro\n"
            "Some content here.\n"
        )
        doc = tmp_path / "course.txt"
        doc.write_text(content)
        course, _ = processor.process_course_document(str(doc))
        assert course.instructor is None

    def test_first_line_without_prefix_becomes_title(self, processor, tmp_path):
        """If the first line lacks 'Course Title:', it's used verbatim as title."""
        content = "My Raw Title\nhttps://example.com\nSomeone\n\nLesson 1: First\nContent."
        doc = tmp_path / "raw.txt"
        doc.write_text(content)
        course, _ = processor.process_course_document(str(doc))
        assert course.title == "My Raw Title"


class TestProcessCourseDocumentLessons:
    def test_correct_lesson_count(self, processor, tmp_path):
        doc = tmp_path / "course.txt"
        doc.write_text(VALID_DOC)
        course, _ = processor.process_course_document(str(doc))
        assert len(course.lessons) == 2

    def test_lesson_titles(self, processor, tmp_path):
        doc = tmp_path / "course.txt"
        doc.write_text(VALID_DOC)
        course, _ = processor.process_course_document(str(doc))
        titles = [l.title for l in course.lessons]
        assert "Getting Started" in titles
        assert "Variables" in titles

    def test_lesson_numbers(self, processor, tmp_path):
        doc = tmp_path / "course.txt"
        doc.write_text(VALID_DOC)
        course, _ = processor.process_course_document(str(doc))
        numbers = [l.lesson_number for l in course.lessons]
        assert 1 in numbers
        assert 2 in numbers

    def test_lesson_link_extracted(self, processor, tmp_path):
        doc = tmp_path / "course.txt"
        doc.write_text(VALID_DOC)
        course, _ = processor.process_course_document(str(doc))
        first_lesson = next(l for l in course.lessons if l.lesson_number == 1)
        assert first_lesson.lesson_link == "https://example.com/python/1"

    def test_lesson_without_link_has_none(self, processor, tmp_path):
        content = (
            "Course Title: Simple\n"
            "Course Link: https://x.com\n"
            "Course Instructor: Bob\n"
            "\n"
            "Lesson 1: No Link Here\n"
            "Content without a lesson link.\n"
        )
        doc = tmp_path / "course.txt"
        doc.write_text(content)
        course, _ = processor.process_course_document(str(doc))
        assert course.lessons[0].lesson_link is None


class TestProcessCourseDocumentChunks:
    def test_chunks_are_created(self, processor, tmp_path):
        doc = tmp_path / "course.txt"
        doc.write_text(VALID_DOC)
        _, chunks = processor.process_course_document(str(doc))
        assert len(chunks) > 0

    def test_chunks_have_correct_course_title(self, processor, tmp_path):
        doc = tmp_path / "course.txt"
        doc.write_text(VALID_DOC)
        _, chunks = processor.process_course_document(str(doc))
        for chunk in chunks:
            assert chunk.course_title == "Python Fundamentals"

    def test_chunks_have_lesson_numbers(self, processor, tmp_path):
        doc = tmp_path / "course.txt"
        doc.write_text(VALID_DOC)
        _, chunks = processor.process_course_document(str(doc))
        lesson_numbers = {c.lesson_number for c in chunks}
        assert lesson_numbers & {1, 2}  # at least one of the lesson numbers present

    def test_chunk_indices_are_sequential_from_zero(self, processor, tmp_path):
        doc = tmp_path / "course.txt"
        doc.write_text(VALID_DOC)
        _, chunks = processor.process_course_document(str(doc))
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_no_lessons_still_creates_chunks(self, processor, tmp_path):
        content = (
            "Course Title: Flat Course\n"
            "Course Link: https://example.com\n"
            "Course Instructor: Bob\n"
            "\n"
            "This is content without any lesson markers at all. "
            "It should still be chunked into the vector store.\n"
        )
        doc = tmp_path / "course.txt"
        doc.write_text(content)
        _, chunks = processor.process_course_document(str(doc))
        assert len(chunks) > 0

    def test_first_chunk_of_lesson_has_context_prefix(self, processor, tmp_path):
        doc = tmp_path / "course.txt"
        doc.write_text(VALID_DOC)
        _, chunks = processor.process_course_document(str(doc))
        # The first chunk of a lesson should contain "Lesson N content:"
        first_chunks = [c for c in chunks if c.lesson_number == 1]
        assert any("Lesson 1" in c.content for c in first_chunks)
