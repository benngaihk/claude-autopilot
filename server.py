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
from fastapi.responses import FileResponse, HTMLResponse
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
