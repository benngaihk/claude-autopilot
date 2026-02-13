#!/usr/bin/env python3
"""Claude Autopilot Dashboard — FastAPI Backend (single file)"""

import asyncio
import json
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.requests import Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from watchfiles import awatch


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(watch_files())
    yield
    task.cancel()


app = FastAPI(title="Claude Autopilot Dashboard", lifespan=lifespan)

# ── Config ───────────────────────────────────────────────────────────────────
CLAUDE_HOME = Path.home() / ".claude"
TASKS_DIR = CLAUDE_HOME / "tasks"
TODOS_DIR = CLAUDE_HOME / "todos"
PROJECT_DIR = Path(__file__).parent
AUTOPILOT_DIR = PROJECT_DIR / ".autopilot"
PROGRESS_FILE = AUTOPILOT_DIR / "progress.md"
LOG_DIR = AUTOPILOT_DIR / "logs"


# ── Helpers ──────────────────────────────────────────────────────────────────
def read_tasks_for_session(session_id: str) -> list[dict]:
    """Read all task JSON files in a session directory."""
    session_dir = TASKS_DIR / session_id
    tasks = []
    if not session_dir.is_dir():
        return tasks
    for f in sorted(session_dir.iterdir()):
        if f.suffix == ".json":
            try:
                data = json.loads(f.read_text())
                if isinstance(data, dict):
                    tasks.append(data)
                elif isinstance(data, list):
                    tasks.extend(data)
            except (json.JSONDecodeError, OSError):
                continue
    return tasks


def read_todos_for_session(session_id: str) -> list[dict]:
    """Read todo JSON files matching a session prefix."""
    todos = []
    if not TODOS_DIR.is_dir():
        return todos
    for f in sorted(TODOS_DIR.iterdir()):
        if f.name.startswith(session_id) and f.suffix == ".json":
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    todos.extend(data)
            except (json.JSONDecodeError, OSError):
                continue
    return todos


def list_sessions() -> list[dict]:
    """List all task sessions with summary info."""
    sessions = []
    if not TASKS_DIR.is_dir():
        return sessions
    for d in sorted(TASKS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir():
            tasks = read_tasks_for_session(d.name)
            total = len(tasks)
            completed = sum(1 for t in tasks if t.get("status") == "completed")
            in_progress = sum(1 for t in tasks if t.get("status") == "in_progress")
            sessions.append({
                "id": d.name,
                "short_id": d.name[:8],
                "total_tasks": total,
                "completed": completed,
                "in_progress": in_progress,
                "pending": total - completed - in_progress,
                "modified": d.stat().st_mtime,
            })
    return sessions


def get_progress() -> str:
    """Read the autopilot progress file."""
    if PROGRESS_FILE.is_file():
        return PROGRESS_FILE.read_text()
    return ""


def list_logs() -> list[dict]:
    """List log files."""
    logs = []
    if not LOG_DIR.is_dir():
        return logs
    for f in sorted(LOG_DIR.iterdir(), reverse=True):
        if f.is_file():
            logs.append({
                "name": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
            })
    return logs


def read_log(filename: str) -> str:
    """Read a specific log file (last 500 lines)."""
    log_path = LOG_DIR / filename
    if not log_path.is_file() or not log_path.is_relative_to(LOG_DIR):
        return ""
    lines = log_path.read_text().splitlines()
    return "\n".join(lines[-500:])


def git_commits_for_task(task_id: str) -> list[dict]:
    """Get git commits mentioning a task ID."""
    try:
        result = subprocess.run(
            ["git", "log", f"--grep={task_id}", "--format=%H|%s|%an|%ai", "-20"],
            capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=10,
        )
        commits = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                commits.append({
                    "hash": parts[0][:8],
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                })
        return commits
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def git_diff_for_commit(commit_hash: str) -> str:
    """Get diff for a specific commit."""
    try:
        result = subprocess.run(
            ["git", "show", "--stat", commit_hash],
            capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=10,
        )
        return result.stdout[:5000]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def read_claude_md() -> str:
    """Read CLAUDE.md file."""
    path = PROJECT_DIR / "CLAUDE.md"
    if path.is_file():
        return path.read_text()
    return ""


# ── REST API ─────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(PROJECT_DIR / "index.html")


@app.get("/api/sessions")
async def api_sessions():
    return list_sessions()


@app.get("/api/tasks/{session_id}")
async def api_tasks(session_id: str):
    return read_tasks_for_session(session_id)


@app.get("/api/todos/{session_id}")
async def api_todos(session_id: str):
    return read_todos_for_session(session_id)


@app.get("/api/progress")
async def api_progress():
    return {"content": get_progress()}


@app.get("/api/logs")
async def api_logs():
    return list_logs()


@app.get("/api/logs/{filename}")
async def api_log_detail(filename: str):
    return {"content": read_log(filename)}


@app.get("/api/commits/{task_id}")
async def api_commits(task_id: str):
    return git_commits_for_task(task_id)


@app.get("/api/diff/{commit_hash}")
async def api_diff(commit_hash: str):
    return {"content": git_diff_for_commit(commit_hash)}


@app.get("/api/claude-md")
async def api_claude_md():
    return {"content": read_claude_md()}


class SaveClaudeMdRequest(BaseModel):
    content: str


@app.post("/api/claude-md")
async def api_save_claude_md(req: SaveClaudeMdRequest):
    path = PROJECT_DIR / "CLAUDE.md"
    path.write_text(req.content)
    return {"ok": True}


# ── Autopilot Control ────────────────────────────────────────────────────────
autopilot_process: subprocess.Popen | None = None


@app.get("/api/autopilot/status")
async def api_autopilot_status():
    state_file = AUTOPILOT_DIR / "state.json"
    state = {}
    if state_file.is_file():
        try:
            state = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    running = autopilot_process is not None and autopilot_process.poll() is None
    return {"running": running, **state}


class AutopilotStartRequest(BaseModel):
    goal: str = ""


@app.post("/api/autopilot/start")
async def api_autopilot_start(req: AutopilotStartRequest):
    global autopilot_process
    if autopilot_process and autopilot_process.poll() is None:
        return {"ok": False, "error": "Already running"}
    goal = req.goal or "Continue from progress.md"
    script = PROJECT_DIR / "autopilot.sh"
    if not script.is_file():
        return {"ok": False, "error": "autopilot.sh not found"}
    env = {**os.environ}
    env.pop("CLAUDECODE", None)  # allow nested launch
    autopilot_process = subprocess.Popen(
        ["bash", str(script), goal],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return {"ok": True, "pid": autopilot_process.pid}


@app.post("/api/autopilot/stop")
async def api_autopilot_stop():
    global autopilot_process
    if autopilot_process and autopilot_process.poll() is None:
        autopilot_process.terminate()
        try:
            autopilot_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            autopilot_process.kill()
        autopilot_process = None
        return {"ok": True}
    return {"ok": False, "error": "Not running"}


# ── Test Runner (SSE stream) ────────────────────────────────────────────────
class TestRunRequest(BaseModel):
    command: str = "pytest"


@app.post("/api/test/run")
async def api_test_run(req: TestRunRequest):
    allowed = ["pytest", "python -m pytest", "playwright test"]
    cmd = req.command.strip()
    if not any(cmd.startswith(a) for a in allowed):
        return {"error": "Command not allowed"}

    async def stream():
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_DIR),
        )
        async for line in proc.stdout:
            yield f"data: {json.dumps({'line': line.decode(errors='replace')})}\n\n"
        await proc.wait()
        yield f"data: {json.dumps({'done': True, 'returncode': proc.returncode})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── WebSocket ────────────────────────────────────────────────────────────────
clients: set[WebSocket] = set()


async def broadcast(message: dict):
    dead = set()
    for ws in clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


async def watch_files():
    """Watch task/todo directories and progress file for changes."""
    watch_paths = []
    if TASKS_DIR.is_dir():
        watch_paths.append(TASKS_DIR)
    if TODOS_DIR.is_dir():
        watch_paths.append(TODOS_DIR)
    if AUTOPILOT_DIR.is_dir():
        watch_paths.append(AUTOPILOT_DIR)

    if not watch_paths:
        return

    try:
        async for changes in awatch(*watch_paths):
            await broadcast({
                "type": "file_change",
                "paths": [str(c[1]) for c in changes],
            })
    except Exception:
        pass


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)


# ── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
