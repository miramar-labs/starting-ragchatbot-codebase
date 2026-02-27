"""Unit tests for Pydantic models (models.py)."""
import pytest
from models import CourseChunk, Course, Lesson


class TestLesson:
    def test_valid_lesson(self):
        lesson = Lesson(lesson_number=1, title="Introduction")
        assert lesson.lesson_number == 1
        assert lesson.title == "Introduction"
        assert lesson.lesson_link is None

    def test_lesson_with_link(self):
        lesson = Lesson(lesson_number=2, title="Setup", lesson_link="https://example.com/2")
        assert lesson.lesson_link == "https://example.com/2"

    def test_lesson_number_required(self):
        with pytest.raises(Exception):
            Lesson(title="Missing number")

    def test_title_required(self):
        with pytest.raises(Exception):
            Lesson(lesson_number=1)

    def test_lesson_number_as_int(self):
        lesson = Lesson(lesson_number=42, title="Deep Dive")
        assert lesson.lesson_number == 42


class TestCourse:
    def test_valid_course_minimal(self):
        course = Course(title="Python 101")
        assert course.title == "Python 101"
        assert course.course_link is None
        assert course.instructor is None
        assert course.lessons == []

    def test_course_with_all_fields(self):
        course = Course(
            title="Python 101",
            course_link="https://example.com",
            instructor="Jane Smith",
        )
        assert course.instructor == "Jane Smith"
        assert course.course_link == "https://example.com"

    def test_title_required(self):
        with pytest.raises(Exception):
            Course()

    def test_lessons_list_is_mutable_per_instance(self):
        """Each Course instance should have its own lessons list."""
        c1 = Course(title="A")
        c2 = Course(title="B")
        c1.lessons.append(Lesson(lesson_number=1, title="L"))
        assert len(c2.lessons) == 0

    def test_course_accepts_lesson_objects(self):
        lesson = Lesson(lesson_number=1, title="Intro")
        course = Course(title="Test Course", lessons=[lesson])
        assert len(course.lessons) == 1
        assert course.lessons[0].title == "Intro"


class TestCourseChunk:
    def test_valid_chunk_minimal(self):
        chunk = CourseChunk(content="Some text", course_title="Python 101", chunk_index=0)
        assert chunk.content == "Some text"
        assert chunk.course_title == "Python 101"
        assert chunk.lesson_number is None
        assert chunk.chunk_index == 0

    def test_chunk_with_lesson_number(self):
        chunk = CourseChunk(
            content="Text", course_title="Course", lesson_number=3, chunk_index=1
        )
        assert chunk.lesson_number == 3

    def test_content_required(self):
        with pytest.raises(Exception):
            CourseChunk(course_title="Course", chunk_index=0)

    def test_course_title_required(self):
        with pytest.raises(Exception):
            CourseChunk(content="Text", chunk_index=0)

    def test_chunk_index_required(self):
        with pytest.raises(Exception):
            CourseChunk(content="Text", course_title="Course")

    def test_chunk_index_is_integer(self):
        chunk = CourseChunk(content="x", course_title="C", chunk_index=99)
        assert chunk.chunk_index == 99
