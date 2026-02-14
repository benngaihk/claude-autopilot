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
    assert "Claude Batch Planner" in r.text


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


def test_api_workspaces_recent(client):
    """GET /api/workspaces/recent should return a list."""
    r = client.get("/api/workspaces/recent")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_api_workspace_tasks(client):
    """GET /api/workspace/tasks should return a list of parsed tasks."""
    r = client.get("/api/workspace/tasks")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_api_workspace_tasks_with_path(client):
    """GET /api/workspace/tasks?path= should handle nonexistent path."""
    r = client.get("/api/workspace/tasks?path=/nonexistent/path")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_api_workspace_goal(client):
    """GET /api/workspace/goal should return goal string."""
    r = client.get("/api/workspace/goal")
    assert r.status_code == 200
    data = r.json()
    assert "goal" in data


def test_api_autopilot_restart_not_running(client):
    """POST /api/autopilot/restart should handle restart when nothing is running."""
    r = client.post("/api/autopilot/restart", json={"goal": "test goal", "workspace": "."})
    assert r.status_code == 200
    data = r.json()
    # It should either start successfully or report an error — both are valid
    assert "ok" in data


# ── Plan & Task API Tests (PRD-driven review board) ──────────────────────────

def test_api_plan_status(client):
    """GET /api/plan/status should return status field."""
    r = client.get("/api/plan/status")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] in ("idle", "planning", "done", "error")


def test_api_plan_tasks_list(client):
    """GET /api/plan/tasks should return a list."""
    r = client.get("/api/plan/tasks")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_api_plan_task_crud(client):
    """Create, read, update a task via file-based store."""
    import json
    import os
    import tempfile

    # Use a temporary workspace directory
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = tmpdir
        tasks_dir = os.path.join(ws, ".autopilot", "tasks")
        os.makedirs(tasks_dir, exist_ok=True)
        # Write a test task JSON
        task = {
            "id": "TASK-TEST-001",
            "subject": "Test task for self-test",
            "prd": "## Objective\nTest PRD content.",
            "status": "pending",
            "dependencies": [],
            "changed_files": [],
            "test_result": None,
            "test_output": "",
            "committed": False,
            "commit_hash": "",
        }
        with open(os.path.join(tasks_dir, "TASK-TEST-001.json"), "w") as f:
            json.dump(task, f)

        # Read tasks
        r = client.get(f"/api/plan/tasks?workspace={tmpdir}")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "TASK-TEST-001"

        # Read single task
        r = client.get(f"/api/plan/tasks/TASK-TEST-001?workspace={tmpdir}")
        assert r.status_code == 200
        data = r.json()
        assert data["subject"] == "Test task for self-test"
        assert data["prd"] == "## Objective\nTest PRD content."

        # Update PRD
        r = client.put(
            f"/api/plan/tasks/TASK-TEST-001?workspace={tmpdir}",
            json={"prd": "## Updated PRD"},
        )
        assert r.status_code == 200
        assert r.json().get("ok") is True

        # Verify update
        r = client.get(f"/api/plan/tasks/TASK-TEST-001?workspace={tmpdir}")
        assert r.json()["prd"] == "## Updated PRD"


def test_api_plan_task_execute_nonexistent(client):
    """POST /api/plan/tasks/{id}/execute should return error for nonexistent task."""
    r = client.post("/api/plan/tasks/TASK-NONEXISTENT/execute?workspace=/tmp")
    assert r.status_code == 200
    data = r.json()
    assert "error" in data


def test_api_plan_task_commit_nonexistent(client):
    """POST /api/plan/tasks/{id}/commit should return error for nonexistent task."""
    r = client.post(
        "/api/plan/tasks/TASK-NONEXISTENT/commit?workspace=/tmp",
        json={"message": "test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "error" in data


def test_api_plan_task_test_nonexistent(client):
    """POST /api/plan/tasks/{id}/test should return error for nonexistent task."""
    r = client.post("/api/plan/tasks/TASK-NONEXISTENT/test?workspace=/tmp")
    assert r.status_code == 200
    data = r.json()
    assert "error" in data


def test_api_plan_task_files_nonexistent(client):
    """GET /api/plan/tasks/{id}/files should return error for nonexistent task."""
    r = client.get("/api/plan/tasks/TASK-NONEXISTENT/files?workspace=/tmp")
    assert r.status_code == 200
    data = r.json()
    assert "error" in data


def test_websocket_connection():
    """WebSocket /ws should accept connection."""
    import websockets.sync.client as ws_client

    with ws_client.connect(f"ws://localhost:8000/ws") as ws:
        # Connection succeeded if we get here
        ws.close()


# ── V2 API Tests (Batch Planner) ─────────────────────────────────────────────

def test_v2_settings_read(client):
    """GET /api/v2/settings should return a dict with default settings."""
    r = client.get("/api/v2/settings")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    assert "max_parallel_tasks" in data


def test_v2_settings_write(client):
    """PUT /api/v2/settings should upsert settings."""
    r = client.put("/api/v2/settings", json={"settings": {"max_parallel_tasks": "5"}})
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    # Verify
    r2 = client.get("/api/v2/settings")
    assert r2.json()["max_parallel_tasks"] == "5"
    # Restore default
    client.put("/api/v2/settings", json={"settings": {"max_parallel_tasks": "3"}})


def test_v2_projects_list(client):
    """GET /api/v2/projects should return a list."""
    r = client.get("/api/v2/projects")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_v2_projects_scan(client):
    """POST /api/v2/projects/scan should return ok with projects list."""
    r = client.post("/api/v2/projects/scan")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert "projects" in data
    assert isinstance(data["projects"], list)


def test_v2_tasks_list(client):
    """GET /api/v2/tasks should return a list."""
    r = client.get("/api/v2/tasks")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


def test_v2_task_not_found(client):
    """GET /api/v2/tasks/{id} should return 404 for nonexistent task."""
    r = client.get("/api/v2/tasks/t-nonexistent")
    assert r.status_code == 404
    data = r.json()
    assert "error" in data


def test_v2_task_approve_not_found(client):
    """POST /api/v2/tasks/{id}/approve should return 404 for nonexistent task."""
    r = client.post("/api/v2/tasks/t-nonexistent/approve")
    assert r.status_code == 404
    data = r.json()
    assert "error" in data


def test_v2_task_reject_not_found(client):
    """POST /api/v2/tasks/{id}/reject should return 404 for nonexistent task."""
    r = client.post("/api/v2/tasks/t-nonexistent/reject")
    assert r.status_code == 404
    data = r.json()
    assert "error" in data


def test_v2_task_replan_not_found(client):
    """POST /api/v2/tasks/{id}/replan should return 404 for nonexistent task."""
    r = client.post("/api/v2/tasks/t-nonexistent/replan")
    assert r.status_code == 404
    data = r.json()
    assert "error" in data


def test_v2_task_cancel_not_found(client):
    """POST /api/v2/tasks/{id}/cancel should return 404 for nonexistent task."""
    r = client.post("/api/v2/tasks/t-nonexistent/cancel")
    assert r.status_code == 404
    data = r.json()
    assert "error" in data


def test_v2_task_diff_not_found(client):
    """GET /api/v2/tasks/{id}/diff should return 404 for nonexistent task."""
    r = client.get("/api/v2/tasks/t-nonexistent/diff")
    assert r.status_code == 404
    data = r.json()
    assert "error" in data


def test_v2_task_merge_not_found(client):
    """POST /api/v2/tasks/{id}/merge should return 404 for nonexistent task."""
    r = client.post("/api/v2/tasks/t-nonexistent/merge")
    assert r.status_code == 404
    data = r.json()
    assert "error" in data


def test_v2_task_delete_not_found(client):
    """DELETE /api/v2/tasks/{id} should return 404 for nonexistent task."""
    r = client.delete("/api/v2/tasks/t-nonexistent")
    assert r.status_code == 404
    data = r.json()
    assert "error" in data


def test_v2_task_lifecycle(client):
    """Test v2 task creation (without Claude, just DB operations)."""
    # First ensure we have at least one project
    client.post("/api/v2/projects/scan")
    projects = client.get("/api/v2/projects").json()
    if not projects:
        # Skip if no projects available
        return

    project_id = projects[0]["id"]

    # Create task (will start planning which will fail without claude, but DB entry is created)
    r = client.post("/api/v2/tasks", json={
        "project_id": project_id,
        "title": "Test task from self-test",
        "description": "This is a test task.",
    })
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    task_id = data["id"]
    assert task_id.startswith("t-")
    assert data["title"] == "Test task from self-test"

    # List tasks — should include our task
    r = client.get("/api/v2/tasks")
    assert r.status_code == 200
    task_ids = [t["id"] for t in r.json()]
    assert task_id in task_ids

    # Filter by project_id
    r = client.get(f"/api/v2/tasks?project_id={project_id}")
    assert r.status_code == 200
    task_ids = [t["id"] for t in r.json()]
    assert task_id in task_ids

    # Get single task
    r = client.get(f"/api/v2/tasks/{task_id}")
    assert r.status_code == 200
    assert r.json()["title"] == "Test task from self-test"

    # Update task
    r = client.put(f"/api/v2/tasks/{task_id}", json={"title": "Updated test task"})
    assert r.status_code == 200
    r2 = client.get(f"/api/v2/tasks/{task_id}")
    assert r2.json()["title"] == "Updated test task"

    # Delete task
    r = client.delete(f"/api/v2/tasks/{task_id}")
    assert r.status_code == 200
    assert r.json().get("ok") is True

    # Verify deletion
    r = client.get(f"/api/v2/tasks/{task_id}")
    assert r.status_code == 404


def test_v2_task_workspace_level(client):
    """Test creating a task without project_id (workspace-level)."""
    r = client.post("/api/v2/tasks", json={
        "title": "Workspace-level test task",
    })
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    task_id = data["id"]
    assert data["project_id"] is None
    assert data["title"] == "Workspace-level test task"

    # Clean up
    r = client.delete(f"/api/v2/tasks/{task_id}")
    assert r.status_code == 200
