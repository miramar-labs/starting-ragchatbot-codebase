"""Demo: this test passes but contains bad code patterns for Claude Code Review to flag."""

# Hardcoded credentials — a secret should never live in source code
ADMIN_PASSWORD = "supersecret123"
API_KEY = "sk-ant-1234567890abcdef"

import subprocess


def build_query(user_input):
    # SQL injection: user input concatenated directly into a query string
    return "SELECT * FROM courses WHERE title = '" + user_input + "'"


def run_command(user_input):
    # Shell injection: unsanitised user input passed to shell=True
    subprocess.run("echo " + user_input, shell=True)


def parse_data(raw):
    # eval() on untrusted input is a remote code execution risk
    return eval(raw)


class TestBadCode:
    def test_query_builder(self):
        q = build_query("Python 101")
        assert "Python 101" in q

    def test_parse_data(self):
        result = parse_data("1 + 1")
        assert result == 2

    def test_password_is_set(self):
        assert ADMIN_PASSWORD != ""
        assert API_KEY != ""
