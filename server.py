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
RECENT_WORKSPACES_FILE = Path.home() / ".claude-autopilot" / "recent.json"

# ── Active workspace (set by /api/autopilot/start) ──────────────────────────
active_workspace: str | None = None
autopilot_start_time: float = 0.0  # timestamp when autopilot was started

# ── Real-time todos captured from stream-json TodoWrite events ──────────────
realtime_todos: list[dict] = []


# ── Workspace Helpers ────────────────────────────────────────────────────────
def parse_progress_tasks(workspace_path: str) -> list[dict]:
    """Parse .autopilot/progress.md in a workspace and return kanban tasks."""
    progress_file = Path(workspace_path) / ".autopilot" / "progress.md"
    if not progress_file.is_file():
        return []
    text = progress_file.read_text()
    tasks = []
    auto_id = 0
    # Map section headers to status
    section_map = {
        "已完成": "completed",
        "进行中": "in_progress",
        "待办": "pending",
        "失败": "failed",
    }
    current_status = None
    for line in text.splitlines():
        stripped = line.strip()
        # Detect section headers like "## 已完成"
        if stripped.startswith("## "):
            header = stripped[3:].strip()
            current_status = section_map.get(header)
            continue
        if current_status is None:
            continue
        # Pattern 1: "- [x] TASK-001: description"
        m = re.match(r"^-\s*\[[ xX]?\]\s*(TASK-\d+)\s*[:：]\s*(.+)", stripped)
        if m:
            tasks.append({
                "id": m.group(1),
                "subject": m.group(2).strip(),
                "status": current_status,
            })
            continue
        # Pattern 2: "- [x] description" (no TASK-xxx prefix, auto-assign ID)
        m2 = re.match(r"^-\s*\[[ xX]?\]\s*(.+)", stripped)
        if m2:
            desc = m2.group(1).strip()
            # Skip placeholder lines
            if desc in ("（无）", "（等待首次 session 拆解）", "无"):
                continue
            auto_id += 1
            # Try to extract TASK-xxx from description text
            tid_match = re.search(r"(TASK-\d+)", desc)
            tid = tid_match.group(1) if tid_match else f"#{auto_id:03d}"
            tasks.append({
                "id": tid,
                "subject": desc,
                "status": current_status,
            })
            continue
        # Pattern 3: plain "- description" (no checkbox)
        m3 = re.match(r"^-\s+(.+)", stripped)
        if m3:
            desc = m3.group(1).strip()
            if desc in ("（无）", "（等待首次 session 拆解）", "无"):
                continue
            auto_id += 1
            tid_match = re.search(r"(TASK-\d+)", desc)
            tid = tid_match.group(1) if tid_match else f"#{auto_id:03d}"
            tasks.append({
                "id": tid,
                "subject": desc,
                "status": current_status,
            })
    return tasks


def read_recent_workspaces() -> list[str]:
    """Read recent workspace paths from ~/.claude-autopilot/recent.json."""
    if not RECENT_WORKSPACES_FILE.is_file():
        return []
    try:
        data = json.loads(RECENT_WORKSPACES_FILE.read_text())
        if isinstance(data, list):
            return [str(p) for p in data if isinstance(p, str)]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def save_recent_workspace(workspace: str):
    """Add a workspace to the recent list (max 10, deduplicated)."""
    recents = read_recent_workspaces()
    # Remove if already exists, then prepend
    recents = [r for r in recents if r != workspace]
    recents.insert(0, workspace)
    recents = recents[:10]
    RECENT_WORKSPACES_FILE.parent.mkdir(parents=True, exist_ok=True)
    RECENT_WORKSPACES_FILE.write_text(json.dumps(recents, indent=2))


def get_latest_session_tasks() -> list[dict]:
    """Read tasks from Claude Code sessions created after autopilot started.

    Claude Code writes task JSON files in real-time via TaskCreate/TaskUpdate.
    We only look at sessions created after autopilot_start_time to avoid
    showing tasks from unrelated sessions.
    """
    if not TASKS_DIR.is_dir() or autopilot_start_time == 0.0:
        return []
    # Find sessions created after autopilot started
    candidates = []
    for d in TASKS_DIR.iterdir():
        if d.is_dir():
            try:
                ctime = d.stat().st_birthtime  # macOS creation time
            except (OSError, AttributeError):
                try:
                    ctime = d.stat().st_mtime
                except OSError:
                    continue
            # Only consider sessions created after autopilot started (with 5s grace)
            if ctime >= autopilot_start_time - 5:
                candidates.append((d, ctime))
    if not candidates:
        return []
    # Use the most recently created session
    candidates.sort(key=lambda x: x[1], reverse=True)
    latest_dir = candidates[0][0]
    # Read task JSONs
    tasks = []
    for f in sorted(latest_dir.iterdir()):
        if f.suffix == ".json":
            try:
                data = json.loads(f.read_text())
                if isinstance(data, dict) and "subject" in data:
                    tasks.append(data)
                elif isinstance(data, list):
                    tasks.extend(t for t in data if isinstance(t, dict) and "subject" in t)
            except (json.JSONDecodeError, OSError):
                continue
    return tasks


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


# ── Workspace API ────────────────────────────────────────────────────────────
@app.post("/api/pick-folder")
async def api_pick_folder():
    """Open native macOS folder picker dialog and return selected path."""
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'POSIX path of (choose folder with prompt "Select workspace folder")'],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            path = result.stdout.strip().rstrip("/")
            return {"path": path}
        return {"path": "", "cancelled": True}
    except subprocess.TimeoutExpired:
        return {"path": "", "error": "Timed out"}
    except FileNotFoundError:
        return {"path": "", "error": "osascript not available"}


@app.get("/api/workspaces/recent")
async def api_recent_workspaces():
    """Get recent workspace paths."""
    return read_recent_workspaces()


@app.get("/api/workspace/tasks")
async def api_workspace_tasks(path: str = ""):
    """Return kanban tasks from the best available real-time source.

    Priority order:
    1. realtime_todos — captured from stream-json TodoWrite events (most real-time)
    2. ~/.claude/tasks/ — Claude Code's native task system (TaskCreate/TaskUpdate)
    3. .autopilot/progress.md — only updated when Claude explicitly writes it
    """
    if not path:
        path = str(PROJECT_DIR)
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        return []
    # Source 1: Real-time todos from stream-json TodoWrite interception
    if realtime_todos:
        return realtime_todos
    # Source 2: Real-time tasks from Claude Code's task system
    realtime_tasks = get_latest_session_tasks()
    if realtime_tasks:
        return [t for t in realtime_tasks if t.get("status") != "deleted"]
    # Source 3: Fallback to progress.md parsing
    return parse_progress_tasks(str(resolved))


@app.get("/api/workspace/goal")
async def api_workspace_goal(path: str = ""):
    """Read the current goal from a workspace's progress.md."""
    if not path:
        path = str(PROJECT_DIR)
    resolved = Path(path).expanduser().resolve()
    progress_file = resolved / ".autopilot" / "progress.md"
    if not progress_file.is_file():
        return {"goal": ""}
    text = progress_file.read_text()
    # Extract goal from "## 目标" section
    in_goal = False
    goal_lines = []
    for line in text.splitlines():
        if line.strip().startswith("## 目标"):
            in_goal = True
            continue
        if in_goal:
            if line.strip().startswith("## "):
                break
            if line.strip():
                goal_lines.append(line.strip())
    return {"goal": " ".join(goal_lines)}


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


def _parse_stream_event(line: str) -> dict | None:
    """Parse a stream-json line from claude -p and extract a human-readable activity event."""
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        # Not JSON — plain text from autopilot.sh banner/status
        if line and not line.startswith("\x1b"):  # skip ANSI-only lines
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
            if clean:
                return {"event": "shell", "summary": clean}
        return None

    msg_type = data.get("type", "")

    # Final result event
    if msg_type == "result":
        cost = data.get("cost_usd", 0)
        turns = data.get("num_turns", 0)
        duration = data.get("duration_ms", 0)
        return {
            "event": "result",
            "summary": f"Session complete — {turns} turns, ${cost:.4f}, {duration / 1000:.0f}s",
            "session_id": data.get("session_id", ""),
        }

    # Assistant message with content blocks
    if msg_type == "assistant":
        global realtime_todos
        msg = data.get("message", {})
        content = msg.get("content", [])
        events = []
        for block in content:
            btype = block.get("type", "")
            if btype == "tool_use":
                tool = block.get("name", "unknown")
                inp = block.get("input", {})
                # Intercept TodoWrite to capture real-time task list
                if tool == "TodoWrite":
                    _capture_todos(inp)
                summary = _tool_summary(tool, inp)
                events.append({"event": "tool_use", "tool": tool, "summary": summary})
            elif btype == "text":
                text = block.get("text", "").strip()
                if text:
                    # Take first 120 chars of text as summary
                    short = text[:120] + ("..." if len(text) > 120 else "")
                    events.append({"event": "text", "summary": short})
        # Return the most interesting event (tool_use > text)
        for e in events:
            if e["event"] == "tool_use":
                return e
        for e in events:
            if e["event"] == "text":
                return e
        return None

    return None


def _capture_todos(inp: dict):
    """Capture todos from a TodoWrite tool call and store as real-time kanban tasks."""
    global realtime_todos
    todos = inp.get("todos", [])
    if not todos:
        return
    captured = []
    for t in todos:
        if not isinstance(t, dict):
            continue
        # TodoWrite format: {id, content, status, priority?}
        tid = str(t.get("id", ""))
        content = t.get("content", "") or t.get("subject", "") or t.get("description", "")
        status = t.get("status", "pending")
        # Normalize status: TodoWrite uses "in_progress", "completed", "pending"
        if status not in ("pending", "in_progress", "completed", "failed"):
            status = "pending"
        captured.append({
            "id": f"TODO-{tid}" if tid and not tid.startswith("TODO") and not tid.startswith("TASK") else (tid or f"TODO-{len(captured)+1}"),
            "subject": content,
            "status": status,
            "description": t.get("description", ""),
        })
    if captured:
        realtime_todos = captured


def _tool_summary(tool: str, inp: dict) -> str:
    """Generate a human-readable summary for a tool call."""
    if tool in ("Read", "read"):
        path = inp.get("file_path", "")
        return f"Reading {Path(path).name}" if path else "Reading file"
    if tool in ("Edit", "edit"):
        path = inp.get("file_path", "")
        return f"Editing {Path(path).name}" if path else "Editing file"
    if tool in ("Write", "write"):
        path = inp.get("file_path", "")
        return f"Writing {Path(path).name}" if path else "Writing file"
    if tool in ("Bash", "bash"):
        cmd = inp.get("command", "")
        short = cmd[:80] + ("..." if len(cmd) > 80 else "")
        return f"Running: {short}" if cmd else "Running command"
    if tool in ("Glob", "glob"):
        return f"Searching: {inp.get('pattern', '?')}"
    if tool in ("Grep", "grep"):
        return f"Grep: {inp.get('pattern', '?')}"
    if tool in ("TaskCreate",):
        return f"Creating task: {inp.get('subject', '?')}"
    if tool in ("TaskUpdate",):
        tid = inp.get("taskId", "?")
        status = inp.get("status", "")
        return f"Task #{tid} → {status}" if status else f"Updating task #{tid}"
    if tool in ("WebFetch",):
        return f"Fetching: {inp.get('url', '?')[:60]}"
    if tool in ("WebSearch",):
        return f"Searching: {inp.get('query', '?')[:60]}"
    if tool in ("Task",):
        desc = inp.get("description", "")
        return f"Spawning agent: {desc}" if desc else "Spawning sub-agent"
    if tool in ("TodoWrite",):
        todos = inp.get("todos", [])
        return f"Updating task list ({len(todos)} items)"
    return f"{tool}"


async def _tail_live_log():
    """Tail the live.log file, parse stream-json events, and broadcast via WebSocket."""
    last_pos = 0
    if LIVE_LOG_FILE.is_file():
        last_pos = LIVE_LOG_FILE.stat().st_size
        text = LIVE_LOG_FILE.read_text(errors="replace")
        lines = text.splitlines(keepends=True)
        autopilot_log_lines.extend(lines[-MAX_LOG_LINES:])
    while True:
        await asyncio.sleep(0.3)  # Faster polling for responsiveness
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
                # Try to parse as stream-json event
                activity = _parse_stream_event(line)
                if activity:
                    activity["timestamp"] = datetime.now().strftime("%H:%M:%S")
                    await broadcast({"type": "autopilot_activity", **activity})
                else:
                    # Fallback: send raw line
                    await broadcast({"type": "autopilot_log", "line": line})
        elif size < last_pos:
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
    workspace: str = ""


@app.post("/api/autopilot/start")
async def api_autopilot_start(req: AutopilotStartRequest):
    global autopilot_pid, active_workspace, autopilot_start_time, realtime_todos
    if autopilot_pid and _is_pid_alive(autopilot_pid):
        return {"ok": False, "error": "Already running"}
    goal = req.goal or "Continue from progress.md"
    workspace = req.workspace or str(PROJECT_DIR)
    workspace_path = Path(workspace).expanduser().resolve()
    if not workspace_path.is_dir():
        return {"ok": False, "error": f"Workspace not found: {workspace}"}
    script = PROJECT_DIR / "autopilot.sh"
    if not script.is_file():
        return {"ok": False, "error": "autopilot.sh not found"}
    active_workspace = str(workspace_path)
    autopilot_start_time = datetime.now().timestamp()
    save_recent_workspace(active_workspace)
    AUTOPILOT_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ}
    env.pop("CLAUDECODE", None)  # allow nested launch
    autopilot_log_lines.clear()
    realtime_todos = []
    # Clear live.log for fresh run
    LIVE_LOG_FILE.write_text("")
    # Launch process detached (start_new_session=True) so uvicorn reload won't kill it
    # Output goes to live.log file instead of pipe
    log_fd = open(LIVE_LOG_FILE, "a")
    proc = subprocess.Popen(
        ["bash", str(script), goal, str(workspace_path)],
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        cwd=str(workspace_path),
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


@app.post("/api/autopilot/restart")
async def api_autopilot_restart(req: AutopilotStartRequest):
    """Stop the current autopilot (if running) and start with a new goal.

    This allows changing goals without returning to the landing page.
    """
    global autopilot_pid, active_workspace, autopilot_start_time

    # 1. Stop current process if running
    if autopilot_pid and _is_pid_alive(autopilot_pid):
        try:
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
        # Brief pause to let process fully terminate
        await asyncio.sleep(1)

    # 2. Start with new goal (reuse start logic)
    goal = req.goal or "Continue from progress.md"
    workspace = req.workspace or str(PROJECT_DIR)
    workspace_path = Path(workspace).expanduser().resolve()
    if not workspace_path.is_dir():
        return {"ok": False, "error": f"Workspace not found: {workspace}"}
    script = PROJECT_DIR / "autopilot.sh"
    if not script.is_file():
        return {"ok": False, "error": "autopilot.sh not found"}
    active_workspace = str(workspace_path)
    autopilot_start_time = datetime.now().timestamp()
    save_recent_workspace(active_workspace)
    AUTOPILOT_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ}
    env.pop("CLAUDECODE", None)
    autopilot_log_lines.clear()
    LIVE_LOG_FILE.write_text("")
    log_fd = open(LIVE_LOG_FILE, "a")
    proc = subprocess.Popen(
        ["bash", str(script), goal, str(workspace_path)],
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        cwd=str(workspace_path),
        env=env,
        start_new_session=True,
    )
    log_fd.close()
    autopilot_pid = proc.pid
    PID_FILE.write_text(str(proc.pid))
    return {"ok": True, "pid": proc.pid}


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
