"""Self-test suite for Claude Autopilot Dashboard.

Tests all API endpoints to verify nothing is broken.
Run: pytest test_self.py -v
"""

import asyncio

import httpx
import pytest

BASE_URL = "http://localhost:8000"


@pytest.fixture
def client():
    """HTTP client for testing."""
    with httpx.Client(base_url=BASE_URL, timeout=10) as c:
        yield c


def test_index_page(client):
    """GET / should return HTML page."""
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Claude Autopilot" in r.text


def test_api_sessions(client):
    """GET /api/sessions should return a list."""
    r = client.get("/api/sessions")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_api_tasks_with_fake_id(client):
    """GET /api/tasks/{id} should return a list (possibly empty)."""
    r = client.get("/api/tasks/nonexistent-session")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_api_progress(client):
    """GET /api/progress should return object with content field."""
    r = client.get("/api/progress")
    assert r.status_code == 200
    data = r.json()
    assert "content" in data


def test_api_logs(client):
    """GET /api/logs should return a list."""
    r = client.get("/api/logs")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_api_claude_md_read(client):
    """GET /api/claude-md should return object with content field."""
    r = client.get("/api/claude-md")
    assert r.status_code == 200
    data = r.json()
    assert "content" in data
    assert len(data["content"]) > 0  # CLAUDE.md should not be empty


def test_api_claude_md_write_and_restore(client):
    """POST /api/claude-md should write and we can restore."""
    # Read original
    original = client.get("/api/claude-md").json()["content"]
    # Write test content
    marker = "\n<!-- self-test marker -->\n"
    r = client.post("/api/claude-md", json={"content": original + marker})
    assert r.status_code == 200
    assert r.json().get("ok") is True
    # Verify write
    updated = client.get("/api/claude-md").json()["content"]
    assert marker in updated
    # Restore original
    r = client.post("/api/claude-md", json={"content": original})
    assert r.status_code == 200
    restored = client.get("/api/claude-md").json()["content"]
    assert restored == original


def test_api_autopilot_status(client):
    """GET /api/autopilot/status should return object with running field."""
    r = client.get("/api/autopilot/status")
    assert r.status_code == 200
    data = r.json()
    assert "running" in data
    assert isinstance(data["running"], bool)


def test_api_autopilot_logs(client):
    """GET /api/autopilot/logs should return object with lines field."""
    r = client.get("/api/autopilot/logs")
    assert r.status_code == 200
    data = r.json()
    assert "lines" in data
    assert isinstance(data["lines"], list)


def test_api_commits_with_fake_id(client):
    """GET /api/commits/{id} should return a list."""
    r = client.get("/api/commits/TASK-999")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_api_todos_with_fake_id(client):
    """GET /api/todos/{id} should return a list."""
    r = client.get("/api/todos/nonexistent-session")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_api_memory_read(client):
    """GET /api/memory should return object with content field."""
    r = client.get("/api/memory")
    assert r.status_code == 200
    data = r.json()
    assert "content" in data


def test_api_memory_write_and_restore(client):
    """POST /api/memory should write and we can restore."""
    original = client.get("/api/memory").json()["content"]
    marker = "\n<!-- memory-test-marker -->\n"
    r = client.post("/api/memory", json={"content": original + marker})
    assert r.status_code == 200
    assert r.json().get("ok") is True
    updated = client.get("/api/memory").json()["content"]
    assert marker in updated
    # Restore
    r = client.post("/api/memory", json={"content": original})
    assert r.status_code == 200


def test_api_daily_logs_list(client):
    """GET /api/memory/daily should return a list."""
    r = client.get("/api/memory/daily")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_api_daily_log_read(client):
    """GET /api/memory/daily/{date} should return content."""
    r = client.get("/api/memory/daily/2099-01-01")
    assert r.status_code == 200
    data = r.json()
    assert "content" in data


def test_api_daily_log_append(client):
    """POST /api/memory/daily should append an entry."""
    r = client.post("/api/memory/daily", json={"entry": "test entry from self-test", "date": "2099-12-31"})
    assert r.status_code == 200
    assert r.json().get("ok") is True
    # Verify it was written
    r2 = client.get("/api/memory/daily/2099-12-31")
    assert "test entry from self-test" in r2.json()["content"]


def test_api_memory_search(client):
    """POST /api/memory/search should return results."""
    # First write something to search
    client.post("/api/memory", json={"content": "# Test Memory\nUnique searchable marker xyzzy123"})
    r = client.post("/api/memory/search", json={"query": "xyzzy123"})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) > 0
    # Clean up
    client.post("/api/memory", json={"content": ""})


def test_api_memory_context(client):
    """GET /api/memory/context should return memory + today + yesterday."""
    r = client.get("/api/memory/context")
    assert r.status_code == 200
    data = r.json()
    assert "memory" in data
    assert "today_log" in data
    assert "yesterday_log" in data


def test_api_summaries_list(client):
    """GET /api/summaries should return a list."""
    r = client.get("/api/summaries")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_api_summary_save_and_read(client):
    """POST /api/summaries should save, GET should read."""
    r = client.post("/api/summaries", json={"session_num": 999, "summary": "Test summary content"})
    assert r.status_code == 200
    assert r.json().get("ok") is True
    r2 = client.get("/api/summaries/session_999.md")
    assert r2.status_code == 200
    assert "Test summary content" in r2.json()["content"]


def test_api_recent_summaries(client):
    """GET /api/summaries/recent/{count} should return content."""
    r = client.get("/api/summaries/recent/3")
    assert r.status_code == 200
    data = r.json()
    assert "content" in data


def test_websocket_connection():
    """WebSocket /ws should accept connection."""
    import websockets.sync.client as ws_client

    with ws_client.connect(f"ws://localhost:8000/ws") as ws:
        # Connection succeeded if we get here
        ws.close()
