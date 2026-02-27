"""Unit tests for SessionManager (session_manager.py)."""
import pytest
from session_manager import SessionManager


class TestSessionCreation:
    def test_create_session_returns_string(self):
        sm = SessionManager()
        assert isinstance(sm.create_session(), str)

    def test_create_session_returns_unique_ids(self):
        sm = SessionManager()
        assert sm.create_session() != sm.create_session()

    def test_create_session_ids_are_incremental(self):
        sm = SessionManager()
        id1 = sm.create_session()
        id2 = sm.create_session()
        num1 = int(id1.split("_")[1])
        num2 = int(id2.split("_")[1])
        assert num2 > num1

    def test_new_session_history_is_none(self):
        sm = SessionManager()
        session_id = sm.create_session()
        assert sm.get_conversation_history(session_id) is None


class TestAddMessage:
    def test_message_appears_in_history(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_message(sid, "user", "Hello there")
        assert "Hello there" in sm.get_conversation_history(sid)

    def test_add_message_to_unknown_session_creates_it(self):
        sm = SessionManager()
        sm.add_message("ghost_session", "user", "Boo")
        history = sm.get_conversation_history("ghost_session")
        assert history is not None
        assert "Boo" in history

    def test_add_exchange_stores_two_messages(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_exchange(sid, "Question?", "Answer!")
        messages = sm.sessions[sid]
        assert len(messages) == 2

    def test_add_exchange_roles_are_correct(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_exchange(sid, "Q", "A")
        assert sm.sessions[sid][0].role == "user"
        assert sm.sessions[sid][1].role == "assistant"


class TestConversationHistory:
    def test_history_has_user_prefix(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_message(sid, "user", "Hi")
        assert "User: Hi" in sm.get_conversation_history(sid)

    def test_history_has_assistant_prefix(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_message(sid, "assistant", "Hello!")
        assert "Assistant: Hello!" in sm.get_conversation_history(sid)

    def test_history_returns_none_for_unknown_session(self):
        sm = SessionManager()
        assert sm.get_conversation_history("does_not_exist") is None

    def test_history_returns_none_for_empty_session(self):
        sm = SessionManager()
        sid = sm.create_session()
        assert sm.get_conversation_history(sid) is None

    def test_history_returns_none_when_id_is_none(self):
        sm = SessionManager()
        assert sm.get_conversation_history(None) is None

    def test_multiple_exchanges_all_appear_in_history(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_exchange(sid, "First question", "First answer")
        sm.add_exchange(sid, "Second question", "Second answer")
        history = sm.get_conversation_history(sid)
        assert "First question" in history
        assert "Second answer" in history

    def test_history_is_newline_separated(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_exchange(sid, "Q", "A")
        history = sm.get_conversation_history(sid)
        assert "\n" in history


class TestHistoryLimit:
    def test_old_messages_trimmed_when_limit_exceeded(self):
        sm = SessionManager(max_history=1)  # 1 turn = 2 messages max
        sid = sm.create_session()
        sm.add_exchange(sid, "Old Q", "Old A")
        sm.add_exchange(sid, "New Q", "New A")
        messages = sm.sessions[sid]
        assert len(messages) == 2
        assert messages[0].content == "New Q"
        assert messages[1].content == "New A"

    def test_messages_within_limit_are_kept(self):
        sm = SessionManager(max_history=3)
        sid = sm.create_session()
        sm.add_exchange(sid, "Q1", "A1")
        sm.add_exchange(sid, "Q2", "A2")
        assert len(sm.sessions[sid]) == 4  # 2 exchanges = 4 messages, within limit

    def test_default_max_history(self):
        sm = SessionManager()
        assert sm.max_history == 5


class TestClearSession:
    def test_clear_removes_all_messages(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_exchange(sid, "Q", "A")
        sm.clear_session(sid)
        assert sm.get_conversation_history(sid) is None

    def test_clear_unknown_session_does_not_raise(self):
        sm = SessionManager()
        sm.clear_session("nonexistent")  # should not raise
