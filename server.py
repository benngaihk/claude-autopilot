#!/usr/bin/env python3
"""Claude Autopilot Dashboard — FastAPI Backend (single file)"""

import asyncio
import json
import os
import re
import signal
import subprocess
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
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
    asyncio.create_task(_reattach_autopilot())
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
LIVE_LOG_FILE = AUTOPILOT_DIR / "live.log"
PID_FILE = AUTOPILOT_DIR / "autopilot.pid"
MEMORY_DIR = AUTOPILOT_DIR / "memory"
MEMORY_FILE = MEMORY_DIR / "MEMORY.md"
SUMMARIES_DIR = AUTOPILOT_DIR / "summaries"


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


# ── Memory Helpers ────────────────────────────────────────────────────────────
def _ensure_memory_dir():
    """Ensure the memory directory exists."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)


def _today_log_path() -> Path:
    return MEMORY_DIR / f"{date.today().isoformat()}.md"


def _yesterday_log_path() -> Path:
    return MEMORY_DIR / f"{(date.today() - timedelta(days=1)).isoformat()}.md"


def read_memory() -> str:
    """Read long-term MEMORY.md."""
    if MEMORY_FILE.is_file():
        return MEMORY_FILE.read_text()
    return ""


def write_memory(content: str):
    """Write long-term MEMORY.md."""
    _ensure_memory_dir()
    MEMORY_FILE.write_text(content)


def read_daily_log(log_date: str | None = None) -> str:
    """Read a daily log. Defaults to today."""
    if log_date is None:
        log_date = date.today().isoformat()
    path = MEMORY_DIR / f"{log_date}.md"
    if path.is_file() and path.is_relative_to(MEMORY_DIR):
        return path.read_text()
    return ""


def append_daily_log(entry: str, log_date: str | None = None):
    """Append an entry to a daily log with timestamp."""
    _ensure_memory_dir()
    if log_date is None:
        log_date = date.today().isoformat()
    path = MEMORY_DIR / f"{log_date}.md"
    if not path.is_relative_to(MEMORY_DIR):
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"\n### {timestamp}\n{entry}\n"
    if not path.is_file():
        # Create with header
        path.write_text(f"# Daily Log — {log_date}\n{line}")
    else:
        with open(path, "a") as f:
            f.write(line)


def list_daily_logs() -> list[dict]:
    """List all daily log files, newest first."""
    logs = []
    if not MEMORY_DIR.is_dir():
        return logs
    for f in sorted(MEMORY_DIR.iterdir(), reverse=True):
        if f.suffix == ".md" and f.name != "MEMORY.md":
            logs.append({
                "date": f.stem,
                "name": f.name,
                "size": f.stat().st_size,
            })
    return logs


def search_memory(query: str) -> list[dict]:
    """Search across MEMORY.md and all daily logs for a query string."""
    results = []
    query_lower = query.lower()
    # Search MEMORY.md
    if MEMORY_FILE.is_file():
        content = MEMORY_FILE.read_text()
        for i, line in enumerate(content.splitlines(), 1):
            if query_lower in line.lower():
                results.append({
                    "file": "MEMORY.md",
                    "line": i,
                    "content": line.strip(),
                })
    # Search daily logs
    if MEMORY_DIR.is_dir():
        for f in sorted(MEMORY_DIR.iterdir(), reverse=True):
            if f.suffix == ".md" and f.name != "MEMORY.md":
                content = f.read_text()
                for i, line in enumerate(content.splitlines(), 1):
                    if query_lower in line.lower():
                        results.append({
                            "file": f.name,
                            "line": i,
                            "content": line.strip(),
                        })
    return results[:100]  # Limit results


def get_context_memory() -> dict:
    """Get memory to inject into session prompts: MEMORY.md + today + yesterday logs."""
    memory_content = read_memory()
    today_log = read_daily_log()
    yesterday_log = read_daily_log((date.today() - timedelta(days=1)).isoformat())
    return {
        "memory": memory_content,
        "today_log": today_log,
        "yesterday_log": yesterday_log,
    }


# ── Summary Helpers (Infinite Context) ────────────────────────────────────────
def save_session_summary(session_num: int, summary: str):
    """Save a session summary for cross-session context continuity."""
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARIES_DIR / f"session_{session_num:03d}.md"
    timestamp = datetime.now().isoformat()
    content = f"# Session {session_num} Summary\n_Generated: {timestamp}_\n\n{summary}\n"
    path.write_text(content)


def list_session_summaries() -> list[dict]:
    """List all session summaries."""
    summaries = []
    if not SUMMARIES_DIR.is_dir():
        return summaries
    for f in sorted(SUMMARIES_DIR.iterdir()):
        if f.suffix == ".md":
            summaries.append({
                "name": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
            })
    return summaries


def read_session_summary(filename: str) -> str:
    """Read a specific session summary."""
    path = SUMMARIES_DIR / filename
    if path.is_file() and path.is_relative_to(SUMMARIES_DIR):
        return path.read_text()
    return ""


def get_recent_summaries(count: int = 3) -> str:
    """Get the N most recent session summaries concatenated."""
    if not SUMMARIES_DIR.is_dir():
        return ""
    files = sorted(SUMMARIES_DIR.iterdir(), reverse=True)
    parts = []
    for f in files[:count]:
        if f.suffix == ".md":
            parts.append(f.read_text())
    return "\n---\n".join(parts)


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


# ── Memory API ────────────────────────────────────────────────────────────────
@app.get("/api/memory")
async def api_memory():
    """Get long-term memory (MEMORY.md)."""
    return {"content": read_memory()}


class SaveMemoryRequest(BaseModel):
    content: str


@app.post("/api/memory")
async def api_save_memory(req: SaveMemoryRequest):
    """Write/replace long-term memory."""
    write_memory(req.content)
    return {"ok": True}


@app.get("/api/memory/daily")
async def api_daily_logs():
    """List all daily logs."""
    return list_daily_logs()


@app.get("/api/memory/daily/{log_date}")
async def api_daily_log(log_date: str):
    """Read a specific daily log."""
    # Validate date format
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", log_date):
        return {"content": "", "error": "Invalid date format"}
    return {"content": read_daily_log(log_date)}


class AppendLogRequest(BaseModel):
    entry: str
    date: str | None = None


@app.post("/api/memory/daily")
async def api_append_daily_log(req: AppendLogRequest):
    """Append an entry to a daily log (defaults to today)."""
    append_daily_log(req.entry, req.date)
    return {"ok": True}


class SearchMemoryRequest(BaseModel):
    query: str


@app.post("/api/memory/search")
async def api_search_memory(req: SearchMemoryRequest):
    """Search across all memory files."""
    return {"results": search_memory(req.query)}


@app.get("/api/memory/context")
async def api_memory_context():
    """Get memory context for prompt injection (MEMORY.md + today + yesterday)."""
    return get_context_memory()


# ── Session Summary API (Infinite Context) ────────────────────────────────────
@app.get("/api/summaries")
async def api_summaries():
    """List all session summaries."""
    return list_session_summaries()


@app.get("/api/summaries/{filename}")
async def api_summary(filename: str):
    """Read a specific session summary."""
    return {"content": read_session_summary(filename)}


class SaveSummaryRequest(BaseModel):
    session_num: int
    summary: str


@app.post("/api/summaries")
async def api_save_summary(req: SaveSummaryRequest):
    """Save a session summary."""
    save_session_summary(req.session_num, req.summary)
    return {"ok": True}


@app.get("/api/summaries/recent/{count}")
async def api_recent_summaries(count: int = 3):
    """Get N most recent session summaries."""
    return {"content": get_recent_summaries(min(count, 10))}


# ── Autopilot Control ────────────────────────────────────────────────────────
autopilot_pid: int | None = None
autopilot_log_lines: list[str] = []
MAX_LOG_LINES = 2000
_tail_task: asyncio.Task | None = None


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


async def _tail_live_log():
    """Tail the live.log file and broadcast new lines via WebSocket."""
    last_pos = 0
    if LIVE_LOG_FILE.is_file():
        last_pos = LIVE_LOG_FILE.stat().st_size
        # Load existing content into buffer
        text = LIVE_LOG_FILE.read_text(errors="replace")
        lines = text.splitlines(keepends=True)
        autopilot_log_lines.extend(lines[-MAX_LOG_LINES:])
    while True:
        await asyncio.sleep(0.5)
        if not LIVE_LOG_FILE.is_file():
            continue
        size = LIVE_LOG_FILE.stat().st_size
        if size > last_pos:
            with open(LIVE_LOG_FILE, "r", errors="replace") as f:
                f.seek(last_pos)
                new_text = f.read()
            last_pos = size
            for line in new_text.splitlines(keepends=True):
                autopilot_log_lines.append(line)
                if len(autopilot_log_lines) > MAX_LOG_LINES:
                    del autopilot_log_lines[:len(autopilot_log_lines) - MAX_LOG_LINES]
                await broadcast({"type": "autopilot_log", "line": line})
        elif size < last_pos:
            # File was truncated/recreated
            last_pos = 0


async def _reattach_autopilot():
    """On server restart, check if an autopilot process is still running."""
    global autopilot_pid, _tail_task
    if PID_FILE.is_file():
        try:
            pid = int(PID_FILE.read_text().strip())
            if _is_pid_alive(pid):
                autopilot_pid = pid
        except (ValueError, OSError):
            pass
    # Always start tailing if log file could exist
    _tail_task = asyncio.create_task(_tail_live_log())


@app.get("/api/autopilot/status")
async def api_autopilot_status():
    global autopilot_pid
    state_file = AUTOPILOT_DIR / "state.json"
    state = {}
    if state_file.is_file():
        try:
            state = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    running = autopilot_pid is not None and _is_pid_alive(autopilot_pid)
    if not running:
        autopilot_pid = None
        if PID_FILE.is_file():
            try:
                PID_FILE.unlink()
            except OSError:
                pass
    return {"running": running, **state}


@app.get("/api/autopilot/logs")
async def api_autopilot_logs(tail: int = 200):
    return {"lines": autopilot_log_lines[-tail:]}


class AutopilotStartRequest(BaseModel):
    goal: str = ""


@app.post("/api/autopilot/start")
async def api_autopilot_start(req: AutopilotStartRequest):
    global autopilot_pid
    if autopilot_pid and _is_pid_alive(autopilot_pid):
        return {"ok": False, "error": "Already running"}
    goal = req.goal or "Continue from progress.md"
    script = PROJECT_DIR / "autopilot.sh"
    if not script.is_file():
        return {"ok": False, "error": "autopilot.sh not found"}
    AUTOPILOT_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ}
    env.pop("CLAUDECODE", None)  # allow nested launch
    autopilot_log_lines.clear()
    # Clear live.log for fresh run
    LIVE_LOG_FILE.write_text("")
    # Launch process detached (start_new_session=True) so uvicorn reload won't kill it
    # Output goes to live.log file instead of pipe
    log_fd = open(LIVE_LOG_FILE, "a")
    proc = subprocess.Popen(
        ["bash", str(script), goal],
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_DIR),
        env=env,
        start_new_session=True,
    )
    log_fd.close()
    autopilot_pid = proc.pid
    PID_FILE.write_text(str(proc.pid))
    return {"ok": True, "pid": proc.pid}


@app.post("/api/autopilot/stop")
async def api_autopilot_stop():
    global autopilot_pid
    if autopilot_pid and _is_pid_alive(autopilot_pid):
        try:
            # Kill the entire process group since we used start_new_session
            os.killpg(autopilot_pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                os.kill(autopilot_pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        autopilot_pid = None
        if PID_FILE.is_file():
            try:
                PID_FILE.unlink()
            except OSError:
                pass
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


# ── Self-Test API ────────────────────────────────────────────────────────────
@app.post("/api/self-test")
async def api_self_test():
    """Run pytest test_self.py -v and stream results via SSE."""
    test_file = PROJECT_DIR / "test_self.py"
    if not test_file.is_file():
        return {"error": "test_self.py not found"}

    async def stream():
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "pytest", str(test_file), "-v", "--tb=short",
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
    """Watch task/todo directories, progress file, memory dir, and project dir for changes."""
    watch_paths = []
    if TASKS_DIR.is_dir():
        watch_paths.append(TASKS_DIR)
    if TODOS_DIR.is_dir():
        watch_paths.append(TODOS_DIR)
    if AUTOPILOT_DIR.is_dir():
        watch_paths.append(AUTOPILOT_DIR)
    if MEMORY_DIR.is_dir():
        watch_paths.append(MEMORY_DIR)
    # Watch project dir for index.html / server.py changes (hot reload)
    watch_paths.append(PROJECT_DIR)

    if not watch_paths:
        return

    try:
        async for changes in awatch(*watch_paths):
            changed_paths = [str(c[1]) for c in changes]
            # Check if index.html changed → trigger UI reload
            if any(p.endswith("index.html") for p in changed_paths):
                await broadcast({"type": "ui_reload"})
            else:
                await broadcast({
                    "type": "file_change",
                    "paths": changed_paths,
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
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_includes=["*.py"],
        reload_excludes=["test_*.py"],
    )
