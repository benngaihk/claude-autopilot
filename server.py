#!/usr/bin/env python3
"""Claude Autopilot Dashboard — FastAPI Backend (single file)"""

import asyncio
import json
import os
import re
import secrets
import signal
import sqlite3
import subprocess
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.requests import Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from watchfiles import awatch


# ── SQLite Database ──────────────────────────────────────────────────────────
DB_PATH: Path | None = None  # set in lifespan after AUTOPILOT_DIR is ensured

def get_db() -> sqlite3.Connection:
    """Return a sqlite3 connection (check_same_thread=False)."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if not exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE,
            main_branch TEXT DEFAULT 'main',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            project_id TEXT REFERENCES projects(id),
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            status TEXT DEFAULT 'created'
                CHECK(status IN ('created','planning','review','executing','done','failed','rejected')),
            plan TEXT,
            plan_raw TEXT,
            execution_result TEXT,
            worktree_path TEXT,
            branch_name TEXT,
            diff_stat TEXT,
            diff_content TEXT,
            error_message TEXT,
            claude_pid INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            planned_at TEXT,
            approved_at TEXT,
            executed_at TEXT,
            merged_at TEXT
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    # Insert default settings if not present
    defaults = {
        "scan_paths": json.dumps([str(PROJECT_DIR.parent)]),
        "max_parallel_tasks": "3",
    }
    for k, v in defaults.items():
        conn.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v)
        )
    conn.commit()
    conn.close()


def _cleanup_orphan_processes():
    """Check tasks with claude_pid set; if process dead, mark failed."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, claude_pid FROM tasks WHERE claude_pid IS NOT NULL"
    ).fetchall()
    for row in rows:
        pid = row["claude_pid"]
        if not _is_pid_alive(pid):
            conn.execute(
                "UPDATE tasks SET claude_pid = NULL, status = 'failed', "
                "error_message = 'Process died unexpectedly', "
                "updated_at = datetime('now') WHERE id = ?",
                (row["id"],),
            )
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app):
    global DB_PATH
    AUTOPILOT_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH = AUTOPILOT_DIR / "autopilot.db"
    init_db()
    _cleanup_orphan_processes()
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

# ── Plan/Task state (PRD-driven review board) ───────────────────────────────
plan_status: str = "idle"  # idle | planning | done | error
plan_error: str = ""
plan_pid: int | None = None
task_processes: dict[str, int] = {}  # task_id → PID
executing_task_id: str | None = None  # currently executing task (serial)

# ── V2 parallel execution semaphore ─────────────────────────────────────────
_v2_semaphore: asyncio.Semaphore | None = None


def _get_v2_semaphore() -> asyncio.Semaphore:
    """Get or create the v2 execution semaphore based on settings."""
    global _v2_semaphore
    if _v2_semaphore is None:
        conn = get_db()
        row = conn.execute(
            "SELECT value FROM settings WHERE key = 'max_parallel_tasks'"
        ).fetchone()
        conn.close()
        limit = int(row["value"]) if row else 3
        _v2_semaphore = asyncio.Semaphore(limit)
    return _v2_semaphore


# ── Task Store Helpers (PRD-driven review board) ────────────────────────────
def _task_store_dir(workspace: str) -> Path:
    """Get/create the .autopilot/tasks/ directory for a workspace."""
    d = Path(workspace) / ".autopilot" / "tasks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_task_store(tasks_dir: Path) -> list[dict]:
    """Read all task JSON files from the tasks directory, sorted by ID."""
    tasks = []
    if not tasks_dir.is_dir():
        return tasks
    for f in sorted(tasks_dir.iterdir()):
        if f.suffix == ".json":
            try:
                data = json.loads(f.read_text())
                if isinstance(data, dict) and "id" in data:
                    tasks.append(data)
            except (json.JSONDecodeError, OSError):
                continue
    return tasks


def _read_task(task_id: str, tasks_dir: Path) -> dict | None:
    """Read a single task by ID."""
    path = tasks_dir / f"{task_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_task(task: dict, tasks_dir: Path):
    """Write a single task to its JSON file."""
    path = tasks_dir / f"{task['id']}.json"
    path.write_text(json.dumps(task, indent=2, ensure_ascii=False))


def _get_changed_files(workspace: str) -> list[str]:
    """Get list of changed files via git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True, text=True, cwd=workspace, timeout=10,
        )
        files = [f for f in result.stdout.strip().splitlines() if f.strip()]
        # Also check staged files
        result2 = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, cwd=workspace, timeout=10,
        )
        staged = [f for f in result2.stdout.strip().splitlines() if f.strip()]
        # Also check untracked files
        result3 = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=workspace, timeout=10,
        )
        untracked = [f for f in result3.stdout.strip().splitlines() if f.strip()]
        return list(dict.fromkeys(files + staged + untracked))  # deduplicate, keep order
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _parse_plan_output(text: str) -> list[dict]:
    """Parse Claude planning output into task list.

    Tries to extract a JSON array of tasks from the output.
    Falls back to extracting markdown task items.
    """
    # Try to find JSON array in the text
    # Look for ```json ... ``` blocks first
    json_match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, list):
                return _normalize_tasks(data)
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON array
    bracket_match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    if bracket_match:
        try:
            data = json.loads(bracket_match.group(0))
            if isinstance(data, list):
                return _normalize_tasks(data)
        except json.JSONDecodeError:
            pass

    # Fallback: parse markdown-style task list
    tasks = []
    task_id = 0
    for line in text.splitlines():
        m = re.match(r"^\s*[-*]\s+\*?\*?(?:TASK-\d+)?:?\s*\*?\*?\s*(.+)", line.strip())
        if m and len(m.group(1).strip()) > 5:
            task_id += 1
            tasks.append({
                "id": f"TASK-{task_id:03d}",
                "subject": m.group(1).strip().rstrip("*"),
                "prd": "",
                "status": "pending",
                "dependencies": [],
                "changed_files": [],
                "test_result": None,
                "test_output": "",
                "committed": False,
                "commit_hash": "",
            })
    return tasks


def _normalize_tasks(data: list) -> list[dict]:
    """Normalize parsed task data into standard format."""
    tasks = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        tid = item.get("id", f"TASK-{i+1:03d}")
        if not tid.startswith("TASK-"):
            tid = f"TASK-{i+1:03d}"
        tasks.append({
            "id": tid,
            "subject": item.get("subject", item.get("title", item.get("name", f"Task {i+1}"))),
            "prd": item.get("prd", item.get("description", "")),
            "status": "pending",
            "dependencies": item.get("dependencies", []),
            "changed_files": [],
            "test_result": None,
            "test_output": "",
            "committed": False,
            "commit_hash": "",
        })
    return tasks


async def _run_planning(goal: str, workspace: str):
    """Run planning phase: claude -p with --max-turns 3, read-only analysis."""
    global plan_status, plan_error, plan_pid

    plan_status = "planning"
    plan_error = ""
    await broadcast({"type": "plan_progress", "message": "Starting planning..."})

    tasks_dir = _task_store_dir(workspace)

    planning_prompt = f"""You are a technical project planner. Analyze the following goal and the codebase at {workspace}.

GOAL: {goal}

OUTPUT FORMAT: Return ONLY a JSON array (wrapped in ```json ... ```) of tasks. Each task object must have:
- "id": "TASK-001", "TASK-002", etc.
- "subject": short title
- "prd": detailed PRD in markdown (## Objective, ## Requirements, ## Acceptance Criteria, ## Technical Notes)
- "dependencies": array of task IDs this depends on (e.g. ["TASK-001"])

RULES:
- Break the goal into 3-8 focused tasks
- Each task should be independently implementable
- Order tasks by dependency (earlier tasks first)
- Write clear, actionable PRDs
- Do NOT write any code, only plan

Return the JSON array now:"""

    try:
        env = {**os.environ}
        env.pop("CLAUDECODE", None)
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", planning_prompt,
            "--max-turns", "3",
            "--output-format", "text",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace,
            env=env,
        )
        plan_pid = proc.pid
        stdout, stderr = await proc.communicate()
        plan_pid = None

        output = stdout.decode(errors="replace")
        if proc.returncode != 0:
            plan_status = "error"
            plan_error = stderr.decode(errors="replace")[:500] or "Planning process failed"
            await broadcast({"type": "plan_error", "message": plan_error})
            return

        # Parse output into tasks
        parsed = _parse_plan_output(output)
        if not parsed:
            plan_status = "error"
            plan_error = "Failed to parse planning output into tasks"
            await broadcast({"type": "plan_error", "message": plan_error})
            return

        # Write tasks to disk
        for task in parsed:
            _write_task(task, tasks_dir)

        plan_status = "done"
        await broadcast({"type": "plan_complete", "tasks": [t["id"] for t in parsed]})

    except Exception as e:
        plan_pid = None
        plan_status = "error"
        plan_error = str(e)
        await broadcast({"type": "plan_error", "message": plan_error})


async def _run_task_execution(task_id: str, workspace: str):
    """Run execution phase for a single task."""
    global executing_task_id

    tasks_dir = _task_store_dir(workspace)
    task = _read_task(task_id, tasks_dir)
    if not task:
        return

    executing_task_id = task_id
    task["status"] = "in_progress"
    _write_task(task, tasks_dir)
    await broadcast({"type": "task_status", "task_id": task_id, "status": "in_progress"})

    # Build context from completed tasks
    all_tasks = _read_task_store(tasks_dir)
    completed_context = ""
    for t in all_tasks:
        if t["status"] == "completed" and t["id"] != task_id:
            completed_context += f"\n- {t['id']}: {t['subject']} (files: {', '.join(t.get('changed_files', []))})"

    execution_prompt = f"""You are a software engineer. Implement the following task in the workspace at {workspace}.

TASK: {task['id']} — {task['subject']}

PRD:
{task.get('prd', 'No PRD provided.')}

{'COMPLETED TASKS (for context):' + completed_context if completed_context else ''}

RULES:
- Implement the task according to the PRD
- Write clean, production-quality code
- Follow existing code patterns and conventions
- Do NOT commit changes (the user will commit manually)
- Do NOT modify unrelated files

Begin implementation:"""

    try:
        env = {**os.environ}
        env.pop("CLAUDECODE", None)
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", execution_prompt,
            "--max-turns", "100",
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace,
            env=env,
        )
        task_processes[task_id] = proc.pid

        # Stream output and broadcast activity
        async for line in proc.stdout:
            decoded = line.decode(errors="replace").strip()
            if not decoded:
                continue
            activity = _parse_stream_event(decoded)
            if activity:
                activity["timestamp"] = datetime.now().strftime("%H:%M:%S")
                activity["task_id"] = task_id
                await broadcast({"type": "task_activity", **activity})

        await proc.wait()
        task_processes.pop(task_id, None)

        # Get changed files
        changed = _get_changed_files(workspace)
        task = _read_task(task_id, tasks_dir)
        if task:
            task["changed_files"] = changed

            # Auto-run tests
            task["status"] = "testing"
            _write_task(task, tasks_dir)
            await broadcast({"type": "task_status", "task_id": task_id, "status": "testing"})
            await broadcast({"type": "task_test", "task_id": task_id})

            test_passed, test_output = await _run_task_tests(workspace)
            task = _read_task(task_id, tasks_dir)
            if task:
                task["test_result"] = "passed" if test_passed else "failed"
                task["test_output"] = test_output
                task["status"] = "completed" if test_passed else "failed"
                _write_task(task, tasks_dir)
                await broadcast({
                    "type": "task_test_done",
                    "task_id": task_id,
                    "passed": test_passed,
                })
                await broadcast({
                    "type": "task_status",
                    "task_id": task_id,
                    "status": task["status"],
                })

    except Exception as e:
        task_processes.pop(task_id, None)
        task = _read_task(task_id, tasks_dir)
        if task:
            task["status"] = "failed"
            task["test_output"] = str(e)
            _write_task(task, tasks_dir)
            await broadcast({"type": "task_status", "task_id": task_id, "status": "failed"})
    finally:
        executing_task_id = None


async def _run_task_tests(workspace: str) -> tuple[bool, str]:
    """Run pytest test_self.py -v and return (passed, output)."""
    test_file = Path(workspace) / "test_self.py"
    if not test_file.is_file():
        return True, "No test_self.py found, skipping tests."

    try:
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "pytest", str(test_file), "-v", "--tb=short",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=workspace,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = stdout.decode(errors="replace")
        return proc.returncode == 0, output
    except asyncio.TimeoutError:
        return False, "Test timed out after 120 seconds."
    except Exception as e:
        return False, f"Test error: {e}"


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


# ── Plan & Task API (PRD-driven review board) ────────────────────────────────
@app.get("/api/plan/status")
async def api_plan_status():
    """Return current planning status."""
    return {"status": plan_status, "error": plan_error}


class PlanGenerateRequest(BaseModel):
    goal: str
    workspace: str


@app.post("/api/plan/generate")
async def api_plan_generate(req: PlanGenerateRequest):
    """Start planning phase — generate task cards + PRDs."""
    global plan_status
    if plan_status == "planning":
        return {"ok": False, "error": "Planning already in progress"}
    workspace_path = Path(req.workspace).expanduser().resolve()
    if not workspace_path.is_dir():
        return {"ok": False, "error": f"Workspace not found: {req.workspace}"}
    save_recent_workspace(str(workspace_path))
    # Reset state
    plan_status = "idle"
    # Start planning in background
    asyncio.create_task(_run_planning(req.goal, str(workspace_path)))
    return {"ok": True}


@app.get("/api/plan/tasks")
async def api_plan_tasks(workspace: str = ""):
    """Read all tasks from .autopilot/tasks/*.json."""
    if not workspace:
        workspace = str(PROJECT_DIR)
    resolved = Path(workspace).expanduser().resolve()
    tasks_dir = resolved / ".autopilot" / "tasks"
    return _read_task_store(tasks_dir)


@app.get("/api/plan/tasks/{task_id}")
async def api_plan_task_detail(task_id: str, workspace: str = ""):
    """Read a single task detail."""
    if not workspace:
        workspace = str(PROJECT_DIR)
    resolved = Path(workspace).expanduser().resolve()
    tasks_dir = resolved / ".autopilot" / "tasks"
    task = _read_task(task_id, tasks_dir)
    if not task:
        return {"error": "Task not found"}
    return task


class TaskUpdateRequest(BaseModel):
    prd: str | None = None
    status: str | None = None
    subject: str | None = None


@app.put("/api/plan/tasks/{task_id}")
async def api_plan_task_update(task_id: str, req: TaskUpdateRequest, workspace: str = ""):
    """Update a task's PRD or status."""
    if not workspace:
        workspace = str(PROJECT_DIR)
    resolved = Path(workspace).expanduser().resolve()
    tasks_dir = resolved / ".autopilot" / "tasks"
    task = _read_task(task_id, tasks_dir)
    if not task:
        return {"error": "Task not found"}
    if req.prd is not None:
        task["prd"] = req.prd
    if req.status is not None:
        task["status"] = req.status
    if req.subject is not None:
        task["subject"] = req.subject
    _write_task(task, tasks_dir)
    await broadcast({"type": "task_status", "task_id": task_id, "status": task["status"]})
    return {"ok": True, "task": task}


@app.post("/api/plan/tasks/{task_id}/execute")
async def api_plan_task_execute(task_id: str, workspace: str = ""):
    """Start execution of a single task."""
    if not workspace:
        workspace = str(PROJECT_DIR)
    resolved = Path(workspace).expanduser().resolve()
    tasks_dir = resolved / ".autopilot" / "tasks"
    task = _read_task(task_id, tasks_dir)
    if not task:
        return {"error": "Task not found"}
    if executing_task_id:
        return {"ok": False, "error": f"Task {executing_task_id} is already executing. Only one task at a time."}
    # Check dependencies
    all_tasks = _read_task_store(tasks_dir)
    task_map = {t["id"]: t for t in all_tasks}
    for dep_id in task.get("dependencies", []):
        dep = task_map.get(dep_id)
        if dep and dep["status"] != "completed":
            return {"ok": False, "error": f"Dependency {dep_id} is not completed yet."}
    # Launch execution in background
    asyncio.create_task(_run_task_execution(task_id, str(resolved)))
    return {"ok": True}


@app.post("/api/plan/tasks/{task_id}/test")
async def api_plan_task_test(task_id: str, workspace: str = ""):
    """Manually trigger tests for a task."""
    if not workspace:
        workspace = str(PROJECT_DIR)
    resolved = Path(workspace).expanduser().resolve()
    tasks_dir = resolved / ".autopilot" / "tasks"
    task = _read_task(task_id, tasks_dir)
    if not task:
        return {"error": "Task not found"}

    task["status"] = "testing"
    _write_task(task, tasks_dir)
    await broadcast({"type": "task_test", "task_id": task_id})

    passed, output = await _run_task_tests(str(resolved))
    task = _read_task(task_id, tasks_dir)
    if task:
        task["test_result"] = "passed" if passed else "failed"
        task["test_output"] = output
        task["status"] = "completed" if passed else "failed"
        _write_task(task, tasks_dir)
        await broadcast({"type": "task_test_done", "task_id": task_id, "passed": passed})
        await broadcast({"type": "task_status", "task_id": task_id, "status": task["status"]})

    return {"ok": True, "passed": passed, "output": output}


class TaskCommitRequest(BaseModel):
    message: str = ""


@app.post("/api/plan/tasks/{task_id}/commit")
async def api_plan_task_commit(task_id: str, req: TaskCommitRequest, workspace: str = ""):
    """Git add + commit for a task's changed files."""
    if not workspace:
        workspace = str(PROJECT_DIR)
    resolved = Path(workspace).expanduser().resolve()
    tasks_dir = resolved / ".autopilot" / "tasks"
    task = _read_task(task_id, tasks_dir)
    if not task:
        return {"error": "Task not found"}

    commit_msg = req.message or f"[{task_id}] feat: {task['subject']}"

    try:
        # Stage changed files
        changed = task.get("changed_files", [])
        if not changed:
            changed = _get_changed_files(str(resolved))
        if not changed:
            return {"ok": False, "error": "No changed files to commit"}

        # git add
        subprocess.run(
            ["git", "add"] + changed,
            capture_output=True, text=True, cwd=str(resolved), timeout=30,
        )
        # git commit
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True, text=True, cwd=str(resolved), timeout=30,
        )
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr[:500] or result.stdout[:500]}

        # Extract commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(resolved), timeout=10,
        )
        commit_hash = hash_result.stdout.strip()

        task = _read_task(task_id, tasks_dir)
        if task:
            task["committed"] = True
            task["commit_hash"] = commit_hash
            _write_task(task, tasks_dir)

        return {"ok": True, "commit_hash": commit_hash}

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/plan/tasks/{task_id}/files")
async def api_plan_task_files(task_id: str, workspace: str = ""):
    """Get changed files for a task."""
    if not workspace:
        workspace = str(PROJECT_DIR)
    resolved = Path(workspace).expanduser().resolve()
    tasks_dir = resolved / ".autopilot" / "tasks"
    task = _read_task(task_id, tasks_dir)
    if not task:
        return {"error": "Task not found"}
    return {"files": task.get("changed_files", [])}


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


# ══════════════════════════════════════════════════════════════════════════════
# ── V2 API Layer ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── V2 Pydantic Models ───────────────────────────────────────────────────────
class V2CreateTaskRequest(BaseModel):
    project_id: str | None = None
    title: str
    description: str = ""


class V2UpdateTaskRequest(BaseModel):
    title: str | None = None
    description: str | None = None


class V2SettingsRequest(BaseModel):
    settings: dict[str, str]


# ── V2 DB helpers ────────────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict."""
    return dict(row)


def _db_get_task(task_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def _db_get_project(project_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def _get_workspace_root() -> str:
    """Get the first scan_path as workspace root directory."""
    conn = get_db()
    row = conn.execute("SELECT value FROM settings WHERE key = 'scan_paths'").fetchone()
    conn.close()
    if row:
        try:
            paths = json.loads(row["value"])
            if paths and isinstance(paths, list):
                return paths[0]
        except (json.JSONDecodeError, IndexError):
            pass
    return str(PROJECT_DIR.parent)


_TASK_COLUMNS = frozenset({
    "title", "description", "status", "plan", "plan_raw", "execution_result",
    "worktree_path", "branch_name", "diff_stat", "diff_content", "error_message",
    "claude_pid", "created_at", "updated_at", "planned_at", "approved_at",
    "executed_at", "merged_at",
})


def _db_update_task(task_id: str, **kwargs):
    """Update arbitrary columns on a task row (column-name whitelist enforced)."""
    if not kwargs:
        return
    kwargs["updated_at"] = datetime.utcnow().isoformat()
    safe = {k: v for k, v in kwargs.items() if k in _TASK_COLUMNS}
    if not safe:
        return
    set_clause = ", ".join(f"{k} = ?" for k in safe)
    values = list(safe.values()) + [task_id]
    conn = get_db()
    conn.execute(f"UPDATE tasks SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()


# ── V2 Project Scanner ──────────────────────────────────────────────────────

def scan_for_projects(scan_paths: list[str], max_depth: int = 2) -> list[dict]:
    """Find git repos by looking for .git directories up to max_depth levels."""
    found = []
    seen_paths: set[str] = set()
    for base in scan_paths:
        base_path = Path(base).expanduser().resolve()
        if not base_path.is_dir():
            continue
        # BFS with depth tracking
        queue: list[tuple[Path, int]] = [(base_path, 0)]
        while queue:
            current, depth = queue.pop(0)
            if depth > max_depth:
                continue
            git_dir = current / ".git"
            if git_dir.exists():
                real_path = str(current)
                if real_path not in seen_paths:
                    seen_paths.add(real_path)
                    found.append({
                        "name": current.name,
                        "path": real_path,
                        "main_branch": detect_main_branch(real_path),
                    })
                continue  # Don't recurse into git repos
            if depth < max_depth:
                try:
                    for child in sorted(current.iterdir()):
                        if child.is_dir() and not child.name.startswith("."):
                            queue.append((child, depth + 1))
                except PermissionError:
                    continue
    return found


def detect_main_branch(project_path: str) -> str:
    """Detect the main branch of a git repo."""
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            capture_output=True, text=True, cwd=project_path, timeout=5,
        )
        if result.returncode == 0:
            ref = result.stdout.strip()
            # refs/remotes/origin/main → main
            return ref.split("/")[-1]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    # Fallback: check if main or master exists
    for branch in ["main", "master"]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
                capture_output=True, text=True, cwd=project_path, timeout=5,
            )
            if result.returncode == 0:
                return branch
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return "main"


# ── V2 Worktree Management ──────────────────────────────────────────────────

async def _create_worktree(project_path: str, task_id: str) -> str:
    """Create git worktree. Returns worktree path."""
    worktree_dir = Path(project_path) / ".worktrees" / task_id
    branch_name = f"task/{task_id}"
    # Ensure .worktrees directory exists
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        "git", "worktree", "add", str(worktree_dir), "-b", branch_name,
        cwd=project_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to create worktree: {stderr.decode(errors='replace')}")
    return str(worktree_dir)


async def _remove_worktree(project_path: str, task_id: str):
    """Remove git worktree and branch."""
    worktree_dir = Path(project_path) / ".worktrees" / task_id
    branch_name = f"task/{task_id}"
    proc = await asyncio.create_subprocess_exec(
        "git", "worktree", "remove", str(worktree_dir), "--force",
        cwd=project_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    proc2 = await asyncio.create_subprocess_exec(
        "git", "branch", "-D", branch_name,
        cwd=project_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc2.communicate()


async def _get_diff(project_path: str, task_id: str, main_branch: str) -> tuple[dict, str]:
    """Get diff between main and task branch. Returns (stat_dict, diff_text)."""
    branch_name = f"task/{task_id}"
    # numstat
    proc = await asyncio.create_subprocess_exec(
        "git", "diff", "--numstat", f"{main_branch}...{branch_name}",
        cwd=project_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    numstat = stdout.decode()
    files_changed = 0
    total_ins = 0
    total_del = 0
    for line in numstat.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            files_changed += 1
            try:
                total_ins += int(parts[0])
                total_del += int(parts[1])
            except ValueError:
                pass
    stat = {"files_changed": files_changed, "total_insertions": total_ins, "total_deletions": total_del}

    # full diff
    proc2 = await asyncio.create_subprocess_exec(
        "git", "diff", f"{main_branch}...{branch_name}",
        cwd=project_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout2, _ = await proc2.communicate()
    diff_text = stdout2.decode(errors="replace")

    # Truncate if exceeds 500KB
    max_diff_size = 500 * 1024
    if len(diff_text) > max_diff_size:
        diff_text = diff_text[:max_diff_size] + "\n\n... [diff truncated, exceeded 500KB] ..."

    return stat, diff_text


async def _merge_branch(project_path: str, task_id: str, main_branch: str, title: str) -> tuple[bool, str]:
    """Merge task branch to main. Returns (success, message)."""
    branch_name = f"task/{task_id}"
    # checkout main
    proc = await asyncio.create_subprocess_exec(
        "git", "checkout", main_branch,
        cwd=project_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    if proc.returncode != 0:
        return False, "Failed to checkout main branch"
    # merge
    proc2 = await asyncio.create_subprocess_exec(
        "git", "merge", branch_name, "--no-ff", "-m", f"Merge {branch_name}: {title}",
        cwd=project_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc2.communicate()
    if proc2.returncode != 0:
        # abort merge
        await asyncio.create_subprocess_exec(
            "git", "merge", "--abort",
            cwd=project_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        return False, stderr.decode(errors="replace")
    return True, stdout.decode(errors="replace")


# ── V2 Planning Function ────────────────────────────────────────────────────

async def _run_v2_planning(task_id: str):
    """Run planning for a v2 task: explore codebase read-only and produce a JSON plan."""
    task = _db_get_task(task_id)
    if not task:
        return

    # Resolve working directory: specific project or workspace root
    if task["project_id"]:
        project = _db_get_project(task["project_id"])
        if not project:
            _db_update_task(task_id, status="failed", error_message="Project not found")
            await broadcast({"type": "task_failed", "task_id": task_id, "error": "Project not found"})
            return
        project_path = project["path"]
    else:
        project_path = _get_workspace_root()

    await broadcast({"type": "task_planning", "task_id": task_id})

    planning_prompt = f"""You are a technical project planner. Analyze the codebase at {project_path} for the following task.

TASK TITLE: {task['title']}
TASK DESCRIPTION: {task['description'] or 'No additional description.'}

Your job is to explore the codebase (READ ONLY — do NOT modify any files) and produce a detailed implementation plan.

OUTPUT FORMAT: Return ONLY a JSON object wrapped in ```json``` markers with this structure:
```json
{{
  "summary": "one line summary of what needs to be done",
  "files_to_modify": [
    {{"path": "relative/path/to/file.py", "action": "modify", "description": "what changes are needed"}},
    {{"path": "relative/path/to/new_file.py", "action": "create", "description": "what this file will contain"}}
  ],
  "approach": "detailed implementation plan in markdown...",
  "risks": ["risk1", "risk2"]
}}
```

RULES:
- Explore the codebase structure first
- List ALL files that need to be modified or created
- action must be one of: "modify", "create", "delete"
- Write a clear, step-by-step approach
- Identify potential risks or edge cases
- Do NOT write any code, only plan
- Do NOT modify any files

Return the JSON plan now:"""

    try:
        env = {**os.environ}
        env.pop("CLAUDECODE", None)
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", planning_prompt,
            "--max-turns", "15",
            "--output-format", "text",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_path,
            env=env,
        )

        stdout, stderr = await proc.communicate()
        output = stdout.decode(errors="replace")

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace")[:500] or "Planning process failed"
            _db_update_task(task_id, status="failed", error_message=err_msg)
            await broadcast({"type": "task_failed", "task_id": task_id, "error": err_msg})
            return

        # Parse JSON from the output
        plan_data = None
        # Try ```json``` block first
        json_match = re.search(r"```json\s*\n(.*?)```", output, re.DOTALL)
        if json_match:
            try:
                plan_data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw JSON object
        if not plan_data:
            obj_match = re.search(r"\{[\s\S]*\"summary\"[\s\S]*\}", output)
            if obj_match:
                try:
                    plan_data = json.loads(obj_match.group(0))
                except json.JSONDecodeError:
                    pass

        if not plan_data:
            err_detail = output.strip()[:1000] if output.strip() else "No output from Claude"
            _db_update_task(
                task_id,
                status="failed",
                plan_raw=output,
                error_message=f"Planning failed — Claude output:\n{err_detail}",
            )
            await broadcast({"type": "task_failed", "task_id": task_id, "error": "Planning did not produce a valid JSON plan"})
            return

        _db_update_task(
            task_id,
            status="review",
            plan=json.dumps(plan_data, ensure_ascii=False),
            plan_raw=output,
            planned_at=datetime.utcnow().isoformat(),
        )
        await broadcast({"type": "task_plan_ready", "task_id": task_id, "plan": plan_data})

    except Exception as e:
        _db_update_task(task_id, status="failed", error_message=str(e))
        await broadcast({"type": "task_failed", "task_id": task_id, "error": str(e)})


# ── V2 Execution Function ───────────────────────────────────────────────────

_V2_EXECUTION_TIMEOUT = 30 * 60  # 30 minutes


async def _run_v2_execution(task_id: str):
    """Execute a v2 task in a git worktree using claude -p."""
    sem = _get_v2_semaphore()
    async with sem:
        task = _db_get_task(task_id)
        if not task:
            return

        # Resolve working directory: specific project or workspace root
        if task["project_id"]:
            project = _db_get_project(task["project_id"])
            if not project:
                _db_update_task(task_id, status="failed", error_message="Project not found")
                await broadcast({"type": "task_failed", "task_id": task_id, "error": "Project not found"})
                return
            project_path = project["path"]
            main_branch = project["main_branch"] or "main"
        else:
            project_path = _get_workspace_root()
            main_branch = "main"

        await broadcast({"type": "task_executing", "task_id": task_id})

        try:
            # Create worktree
            worktree_path = await _create_worktree(project_path, task_id)
            branch_name = f"task/{task_id}"
            _db_update_task(task_id, worktree_path=worktree_path, branch_name=branch_name)

            # Parse plan for context
            plan_text = ""
            if task["plan"]:
                try:
                    plan_data = json.loads(task["plan"])
                    plan_text = f"""
PLAN SUMMARY: {plan_data.get('summary', '')}

APPROACH:
{plan_data.get('approach', '')}

FILES TO MODIFY:
{json.dumps(plan_data.get('files_to_modify', []), indent=2)}
"""
                except json.JSONDecodeError:
                    plan_text = task["plan"]

            execution_prompt = f"""You are a software engineer. Implement the following task in the workspace at {worktree_path}.

TASK: {task['title']}
DESCRIPTION: {task['description'] or 'No additional description.'}

{plan_text}

RULES:
- Implement the task according to the plan
- Write clean, production-quality code
- Follow existing code patterns and conventions
- Commit your changes with a descriptive message
- Do NOT modify unrelated files

Begin implementation:"""

            env = {**os.environ}
            env.pop("CLAUDECODE", None)
            proc = await asyncio.create_subprocess_exec(
                "claude", "-p", execution_prompt,
                "--max-turns", "100",
                "--output-format", "stream-json",
                "--dangerously-skip-permissions",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=worktree_path,
                env=env,
            )
            _db_update_task(task_id, claude_pid=proc.pid)

            # Stream output with timeout
            try:
                async def _stream_output():
                    async for line in proc.stdout:
                        decoded = line.decode(errors="replace").strip()
                        if not decoded:
                            continue
                        activity = _parse_stream_event(decoded)
                        if activity:
                            activity["timestamp"] = datetime.now().strftime("%H:%M:%S")
                            activity["task_id"] = task_id
                            await broadcast({"type": "task_activity", **activity})

                await asyncio.wait_for(_stream_output(), timeout=_V2_EXECUTION_TIMEOUT)
                await proc.wait()
            except asyncio.TimeoutError:
                # Kill the process on timeout
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                _db_update_task(
                    task_id,
                    status="failed",
                    claude_pid=None,
                    error_message=f"Execution timed out after {_V2_EXECUTION_TIMEOUT // 60} minutes",
                )
                await broadcast({"type": "task_failed", "task_id": task_id, "error": "Execution timed out"})
                return

            _db_update_task(task_id, claude_pid=None)

            # Generate diff
            stat, diff_text = await _get_diff(project_path, task_id, main_branch)

            _db_update_task(
                task_id,
                status="done",
                diff_stat=json.dumps(stat),
                diff_content=diff_text,
                executed_at=datetime.utcnow().isoformat(),
            )
            await broadcast({"type": "task_done", "task_id": task_id, "diff_stat": stat})

        except Exception as e:
            _db_update_task(
                task_id,
                status="failed",
                claude_pid=None,
                error_message=str(e),
            )
            await broadcast({"type": "task_failed", "task_id": task_id, "error": str(e)})


# ── V2 Settings API ─────────────────────────────────────────────────────────

@app.get("/api/v2/settings")
async def api_v2_settings():
    """Return all settings as a dict."""
    conn = get_db()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}


@app.put("/api/v2/settings")
async def api_v2_settings_update(req: V2SettingsRequest):
    """Upsert settings."""
    global _v2_semaphore
    conn = get_db()
    for k, v in req.settings.items():
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (k, v),
        )
    conn.commit()
    conn.close()
    # Reset semaphore if max_parallel_tasks changed
    if "max_parallel_tasks" in req.settings:
        _v2_semaphore = None
    return {"ok": True}


# ── V2 Projects API ─────────────────────────────────────────────────────────

@app.get("/api/v2/projects")
async def api_v2_projects():
    """List all projects from DB."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM projects ORDER BY name").fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


@app.post("/api/v2/projects/scan")
async def api_v2_projects_scan():
    """Scan for projects and upsert into DB."""
    conn = get_db()
    # Get scan paths from settings
    row = conn.execute("SELECT value FROM settings WHERE key = 'scan_paths'").fetchone()
    conn.close()
    scan_paths = [str(PROJECT_DIR.parent)]
    if row:
        try:
            scan_paths = json.loads(row["value"])
        except json.JSONDecodeError:
            pass

    projects = scan_for_projects(scan_paths)

    conn = get_db()
    try:
        # Remove projects no longer in scan paths, then upsert current ones
        scanned_paths = {p["path"] for p in projects}
        existing = conn.execute("SELECT id, path FROM projects").fetchall()
        for row in existing:
            if row["path"] not in scanned_paths:
                conn.execute("DELETE FROM projects WHERE id = ?", (row["id"],))
        for p in projects:
            pid = "p-" + secrets.token_hex(4)
            conn.execute(
                "INSERT INTO projects (id, name, path, main_branch) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET name = excluded.name, main_branch = excluded.main_branch",
                (pid, p["name"], p["path"], p["main_branch"]),
            )
        conn.commit()
        # Return updated list
        rows = conn.execute("SELECT * FROM projects ORDER BY name").fetchall()
        return {"ok": True, "projects": [_row_to_dict(r) for r in rows]}
    finally:
        conn.close()


# ── V2 Task API ─────────────────────────────────────────────────────────────

@app.post("/api/v2/tasks")
async def api_v2_task_create(req: V2CreateTaskRequest):
    """Create a new v2 task and immediately launch background planning."""
    # Validate project exists if specified
    if req.project_id:
        project = _db_get_project(req.project_id)
        if not project:
            return JSONResponse(status_code=404, content={"error": "Project not found"})

    task_id = "t-" + secrets.token_hex(4)
    conn = get_db()
    conn.execute(
        "INSERT INTO tasks (id, project_id, title, description, status) "
        "VALUES (?, ?, ?, ?, 'planning')",
        (task_id, req.project_id, req.title, req.description),
    )
    conn.commit()
    conn.close()

    task = _db_get_task(task_id)
    await broadcast({"type": "task_created", "task": task})

    # Launch planning in background
    asyncio.create_task(_run_v2_planning(task_id))

    return task


@app.get("/api/v2/tasks")
async def api_v2_task_list(project_id: str | None = None, status: str | None = None):
    """List v2 tasks with optional filters."""
    conn = get_db()
    query = "SELECT * FROM tasks WHERE 1=1"
    params: list = []
    if project_id:
        query += " AND project_id = ?"
        params.append(project_id)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


@app.get("/api/v2/tasks/{task_id}")
async def api_v2_task_detail(task_id: str):
    """Get a single v2 task."""
    task = _db_get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return task


@app.put("/api/v2/tasks/{task_id}")
async def api_v2_task_update(task_id: str, req: V2UpdateTaskRequest):
    """Update title/description of a v2 task."""
    task = _db_get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    updates = {}
    if req.title is not None:
        updates["title"] = req.title
    if req.description is not None:
        updates["description"] = req.description
    if updates:
        _db_update_task(task_id, **updates)
    return _db_get_task(task_id)


@app.delete("/api/v2/tasks/{task_id}")
async def api_v2_task_delete(task_id: str):
    """Delete a v2 task. Clean up worktree and kill process if needed."""
    task = _db_get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    # Kill claude process if running
    if task["claude_pid"] and _is_pid_alive(task["claude_pid"]):
        try:
            os.kill(task["claude_pid"], signal.SIGTERM)
            await asyncio.sleep(1)  # wait for graceful shutdown before worktree cleanup
        except (OSError, ProcessLookupError):
            pass

    # Remove worktree if exists
    if task["worktree_path"] and Path(task["worktree_path"]).exists():
        project = _db_get_project(task["project_id"])
        if project:
            try:
                await _remove_worktree(project["path"], task_id)
            except Exception:
                pass

    conn = get_db()
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/api/v2/tasks/{task_id}/approve")
async def api_v2_task_approve(task_id: str):
    """Approve plan, create worktree, start execution."""
    task = _db_get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    if task["status"] not in ("review", "failed"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Cannot approve task in status '{task['status']}'"},
        )

    _db_update_task(
        task_id,
        status="executing",
        approved_at=datetime.utcnow().isoformat(),
    )

    await broadcast({"type": "task_approved", "task_id": task_id})

    # Launch execution in background
    asyncio.create_task(_run_v2_execution(task_id))

    return {"ok": True, "task": _db_get_task(task_id)}


@app.post("/api/v2/tasks/{task_id}/reject")
async def api_v2_task_reject(task_id: str):
    """Reject a task's plan."""
    task = _db_get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    _db_update_task(task_id, status="rejected")
    return {"ok": True, "task": _db_get_task(task_id)}


@app.post("/api/v2/tasks/{task_id}/replan")
async def api_v2_task_replan(task_id: str):
    """Reset task to planning and re-run the planner."""
    task = _db_get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    _db_update_task(
        task_id,
        status="planning",
        plan=None,
        plan_raw=None,
        planned_at=None,
        error_message=None,
    )

    asyncio.create_task(_run_v2_planning(task_id))
    return {"ok": True, "task": _db_get_task(task_id)}


@app.post("/api/v2/tasks/{task_id}/cancel")
async def api_v2_task_cancel(task_id: str):
    """Cancel a task: kill process, remove worktree, set failed."""
    task = _db_get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    # Kill claude process if running
    if task["claude_pid"] and _is_pid_alive(task["claude_pid"]):
        try:
            os.kill(task["claude_pid"], signal.SIGTERM)
            await asyncio.sleep(1)  # wait for graceful shutdown before worktree cleanup
        except (OSError, ProcessLookupError):
            pass

    # Remove worktree if exists
    if task["worktree_path"] and Path(task["worktree_path"]).exists():
        project = _db_get_project(task["project_id"])
        if project:
            try:
                await _remove_worktree(project["path"], task_id)
            except Exception:
                pass

    _db_update_task(
        task_id,
        status="failed",
        claude_pid=None,
        error_message="Cancelled by user",
    )
    return {"ok": True, "task": _db_get_task(task_id)}


@app.get("/api/v2/tasks/{task_id}/diff")
async def api_v2_task_diff(task_id: str):
    """Return diff_stat and diff_content for a task."""
    task = _db_get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    diff_stat = None
    if task["diff_stat"]:
        try:
            diff_stat = json.loads(task["diff_stat"])
        except json.JSONDecodeError:
            diff_stat = task["diff_stat"]

    return {
        "task_id": task_id,
        "diff_stat": diff_stat,
        "diff_content": task["diff_content"],
    }


@app.post("/api/v2/tasks/{task_id}/merge")
async def api_v2_task_merge(task_id: str):
    """Merge task branch to main, clean up worktree."""
    task = _db_get_task(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    if task["status"] != "done":
        return JSONResponse(
            status_code=400,
            content={"error": f"Cannot merge task in status '{task['status']}'. Must be 'done'."},
        )

    project = _db_get_project(task["project_id"])
    if not project:
        return JSONResponse(status_code=404, content={"error": "Project not found"})

    main_branch = project["main_branch"] or "main"

    success, message = await _merge_branch(
        project["path"], task_id, main_branch, task["title"]
    )

    if not success:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"Merge failed: {message}"},
        )

    # Clean up worktree
    try:
        await _remove_worktree(project["path"], task_id)
    except Exception:
        pass

    _db_update_task(
        task_id,
        merged_at=datetime.utcnow().isoformat(),
        worktree_path=None,
    )

    await broadcast({"type": "task_merged", "task_id": task_id})

    return {"ok": True, "message": message, "task": _db_get_task(task_id)}


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
