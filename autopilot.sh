#!/usr/bin/env bash
# ==============================================================================
# Claude Autopilot Core — 最小内核
# 让 Claude Code 长时间自动运行，session 断了能恢复
#
# 用法:
#   chmod +x autopilot.sh
#   ./autopilot.sh "构建 Autopilot UI 看板系统"
#
# 原理:
#   外层 bash 循环 → 调用 claude -p → session 结束 → 读进度 → 重启
#   Claude Code 内部用 Agent Teams 自主编排多 Agent
# ==============================================================================

set -euo pipefail

# ── 解除嵌套限制 ─────────────────────────────────────────────────────────────
# 从 Dashboard (server.py) 或另一个 Claude Code 中启动时需要清除此变量
unset CLAUDECODE 2>/dev/null || true

# ── 配置 ──────────────────────────────────────────────────────────────────────
PROJECT_DIR="${2:-.}"                    # 项目目录，默认当前目录
MAX_SESSIONS="${3:-10}"                  # 最多重启几轮
MAX_TURNS="${4:-200}"                    # 每轮 session 最大交互轮次
COOLDOWN="${5:-15}"                      # 两轮之间冷却秒数
AUTOPILOT_DIR="$PROJECT_DIR/.autopilot"
PROGRESS_FILE="$AUTOPILOT_DIR/progress.md"
LOG_DIR="$AUTOPILOT_DIR/logs"
STATE_FILE="$AUTOPILOT_DIR/state.json"
MEMORY_DIR="$AUTOPILOT_DIR/memory"
MEMORY_FILE="$MEMORY_DIR/MEMORY.md"
SUMMARIES_DIR="$AUTOPILOT_DIR/summaries"

# ── 颜色 ──────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── 初始化 ────────────────────────────────────────────────────────────────────
init() {
    local goal="$1"
    mkdir -p "$AUTOPILOT_DIR" "$LOG_DIR" "$MEMORY_DIR" "$SUMMARIES_DIR"

    # 保存当前目标，用于检测目标是否变更
    local goal_file="$AUTOPILOT_DIR/current_goal.txt"
    local prev_goal=""
    if [ -f "$goal_file" ]; then
        prev_goal=$(cat "$goal_file")
    fi

    # 如果目标变了，或者进度文件标记为已完成 → 重置进度
    local need_reset=false
    if [ "$goal" != "$prev_goal" ] && [ -n "$goal" ] && [ "$goal" != "Continue from progress.md" ]; then
        need_reset=true
        echo -e "${YELLOW}⚠ 检测到新目标，重置进度文件...${NC}"
    elif [ -f "$PROGRESS_FILE" ] && grep -q "## 状态: 已完成" "$PROGRESS_FILE"; then
        need_reset=true
        echo -e "${YELLOW}⚠ 上一轮已完成，重置进度文件...${NC}"
    fi

    if [ "$need_reset" = true ]; then
        # 归档旧进度文件
        if [ -f "$PROGRESS_FILE" ]; then
            local archive_name="progress_$(date +%Y%m%d_%H%M%S).md"
            cp "$PROGRESS_FILE" "$AUTOPILOT_DIR/$archive_name"
            echo -e "${CYAN}  旧进度已归档为 $archive_name${NC}"
        fi
    fi

    # 保存当前目标
    echo "$goal" > "$goal_file"

    # 初始化/重置状态文件
    if [ ! -f "$STATE_FILE" ] || [ "$need_reset" = true ]; then
        cat > "$STATE_FILE" << 'EOF'
{
  "session_count": 0,
  "total_tasks_completed": 0,
  "status": "idle",
  "started_at": null,
  "last_session_at": null
}
EOF
    fi

    # 初始化/重置进度文件
    if [ ! -f "$PROGRESS_FILE" ] || [ "$need_reset" = true ]; then
        cat > "$PROGRESS_FILE" << EOF
# Autopilot 进度记录

## 状态: 未开始

## 目标
${goal}

## 已完成
（无）

## 进行中
（无）

## 待办
（等待首次 session 拆解）
EOF
    fi
}

# ── 加载记忆上下文 ────────────────────────────────────────────────────────────
load_memory_context() {
    local memory_context=""

    # 长期记忆
    if [ -f "$MEMORY_FILE" ]; then
        local mem_content
        mem_content=$(head -200 "$MEMORY_FILE")
        if [ -n "$mem_content" ]; then
            memory_context="${memory_context}
### 长期记忆 (MEMORY.md)
${mem_content}
"
        fi
    fi

    # 今日日志
    local today
    today=$(date +%Y-%m-%d)
    local today_log="$MEMORY_DIR/${today}.md"
    if [ -f "$today_log" ]; then
        local today_content
        today_content=$(tail -100 "$today_log")
        memory_context="${memory_context}
### 今日日志 (${today})
${today_content}
"
    fi

    # 昨日日志
    local yesterday
    yesterday=$(date -v-1d +%Y-%m-%d 2>/dev/null || date -d "yesterday" +%Y-%m-%d 2>/dev/null || echo "")
    if [ -n "$yesterday" ]; then
        local yesterday_log="$MEMORY_DIR/${yesterday}.md"
        if [ -f "$yesterday_log" ]; then
            local yesterday_content
            yesterday_content=$(tail -50 "$yesterday_log")
            memory_context="${memory_context}
### 昨日日志 (${yesterday})
${yesterday_content}
"
        fi
    fi

    # 最近 session 摘要
    if [ -d "$SUMMARIES_DIR" ]; then
        local recent_summaries
        recent_summaries=$(ls -t "$SUMMARIES_DIR"/*.md 2>/dev/null | head -3)
        if [ -n "$recent_summaries" ]; then
            memory_context="${memory_context}
### 最近 Session 摘要
"
            for f in $recent_summaries; do
                memory_context="${memory_context}$(cat "$f")
---
"
            done
        fi
    fi

    echo "$memory_context"
}

# ── 构建 Prompt ──────────────────────────────────────────────────────────────
build_prompt() {
    local goal="$1"
    local session_num="$2"
    local progress=""
    local memory_context=""

    if [ -f "$PROGRESS_FILE" ]; then
        progress=$(cat "$PROGRESS_FILE")
    fi

    # 加载记忆上下文
    memory_context=$(load_memory_context)

    cat << PROMPT
## 目标
${goal}

## 你是谁
你是一个自主编排的 AI 开发团队 Team Lead。
你必须独立完成目标，不要问我问题，自己做决策。

## 当前 Session
这是第 ${session_num} 轮 session（最多 ${MAX_SESSIONS} 轮）。
每轮 session 有上下文窗口限制，做不完没关系，把进度写好，下轮继续。

## 上次进度
${progress}

## 记忆上下文
${memory_context}

## 实时任务追踪（重要！）

**你必须使用 TaskCreate 和 TaskUpdate 工具来管理任务。** 这样 Dashboard 才能实时显示进度。

工作流:
1. 分析目标后，**立即** 用 TaskCreate 创建所有任务（subject + description + activeForm）
2. 开始做某个任务前，用 TaskUpdate 把它设为 in_progress
3. 做完后，用 TaskUpdate 把它设为 completed
4. 如果有依赖关系，用 TaskUpdate 的 addBlockedBy/addBlocks 设置

示例:
\`\`\`
TaskCreate({ subject: "创建数据库 schema", description: "设计用户表...", activeForm: "创建数据库 schema" })
TaskUpdate({ taskId: "1", status: "in_progress" })
// ... 执行任务 ...
TaskUpdate({ taskId: "1", status: "completed" })
\`\`\`

## 工作流程

### 首次 Session（进度文件为空时）
1. 分析目标，拆解为具体任务
2. **立即用 TaskCreate 创建所有任务**（这样 Dashboard 能实时显示）
3. 确定任务依赖关系，用 TaskUpdate 设置 addBlockedBy
4. 同步写任务清单到 ${PROGRESS_FILE}
5. 开始执行第一批无依赖的任务
6. 如果任务适合并行，用 Agent Teams

### 后续 Session（有进度时）
1. 读取 ${PROGRESS_FILE}，了解上次做到哪里
2. 用 TaskCreate 重新创建未完成的任务（新 session 需要重新创建）
3. 继续未完成的任务
4. 更新 ${PROGRESS_FILE}

### 每次 Session 结束前（必须做）
1. git add + git commit 所有变更
2. 更新 ${PROGRESS_FILE}
3. **记忆管理**: 将本次 session 的关键决策和发现写入记忆系统:
   - 重要决策、偏好、架构约定 → 写入 \`${MEMORY_FILE}\`
   - 当日工作内容、下一步计划 → 追加到 \`${MEMORY_DIR}/$(date +%Y-%m-%d).md\`
   - 提供本次 session 的简短摘要（2-3段），保存到 \`${SUMMARIES_DIR}/session_$(printf '%03d' $session_num).md\`

进度文件格式:
\`\`\`markdown
# Autopilot 进度记录

## 状态: 进行中 / 已完成

## 已完成
- [x] TASK-001: 具体描述 (哪些文件)
- [x] TASK-002: 具体描述

## 进行中
- [ ] TASK-003: 当前做到哪一步，下一步要做什么

## 待办
- [ ] TASK-004: 描述
- [ ] TASK-005: 描述（依赖 TASK-003）

## 关键上下文（给下一轮 session 的备注）
- 数据库用的 PostgreSQL，schema 在 xxx 文件
- API base URL: http://localhost:8080
- 前端 dev server: http://localhost:5173
- 注意: xxx 模块有个已知问题需要先解决
\`\`\`

## 规则
- 不要问我问题，自己做所有决策
- 遇到不确定的，选最合理的方案直接做
- 每完成一个任务就 git commit，格式: [TASK-ID] type: description
- 绝对不要 git push --force
- 绝对不要删除 .env、package.json、pom.xml 等关键文件
- 如果接近上下文限制，立即停下来：**先保存记忆**，再更新进度文件

PROMPT
}

# ── 检查是否完成 ─────────────────────────────────────────────────────────────
check_done() {
    if [ -f "$PROGRESS_FILE" ]; then
        if grep -q "## 状态: 已完成" "$PROGRESS_FILE"; then
            return 0
        fi
    fi
    return 1
}

# ── 运行一轮 Session ─────────────────────────────────────────────────────────
run_session() {
    local goal="$1"
    local session_num="$2"
    local log_file="$LOG_DIR/session_$(printf '%03d' $session_num)_$(date +%Y%m%d_%H%M%S).log"
    local prompt

    prompt=$(build_prompt "$goal" "$session_num")

    echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  Session ${session_num}/${MAX_SESSIONS}  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"

    # 更新状态
    python3 -c "
import json, datetime
s = json.load(open('$STATE_FILE'))
s['session_count'] = $session_num
s['status'] = 'running'
s['last_session_at'] = datetime.datetime.now().isoformat()
if not s['started_at']:
    s['started_at'] = s['last_session_at']
json.dump(s, open('$STATE_FILE', 'w'), indent=2)
" 2>/dev/null || true

    echo -e "${GREEN}▶ 启动 Claude Code...${NC}"

    set +e
    # stream-json 模式: 每个事件一行 JSON，Dashboard 可以实时解析
    # --verbose 是 stream-json 的必要前置条件
    claude -p "$prompt" \
        --max-turns "$MAX_TURNS" \
        --dangerously-skip-permissions \
        --output-format stream-json \
        --verbose
    local exit_code=$?
    set -e

    echo ""
    echo -e "${YELLOW}Session ${session_num} 结束 (exit code: ${exit_code})${NC}"

    # 记录日志
    echo "---" >> "$log_file"
    echo "Exit code: $exit_code" >> "$log_file"
    echo "Ended at: $(date -Iseconds)" >> "$log_file"

    # 自动追加每日日志条目
    local today
    today=$(date +%Y-%m-%d)
    local today_log="$MEMORY_DIR/${today}.md"
    local timestamp
    timestamp=$(date +%H:%M:%S)
    local log_entry="
### ${timestamp} — Session ${session_num} 自动记录
- Exit code: ${exit_code}
- 日志: $(basename "$log_file")
"
    if [ ! -f "$today_log" ]; then
        echo "# Daily Log — ${today}" > "$today_log"
    fi
    echo "$log_entry" >> "$today_log"

    return $exit_code
}

# ── 主循环 ────────────────────────────────────────────────────────────────────
main() {
    local goal="${1:-请告诉我要做什么}"

    echo -e "${GREEN}"
    echo "  ╔═══════════════════════════════════════════╗"
    echo "  ║       ⚡ Claude Autopilot Core ⚡         ║"
    echo "  ║       让 Claude Code 长时间自动运行        ║"
    echo "  ╚═══════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  目标: ${CYAN}${goal}${NC}"
    echo -e "  项目: ${CYAN}${PROJECT_DIR}${NC}"
    echo -e "  最大轮次: ${CYAN}${MAX_SESSIONS}${NC}"
    echo ""

    init "$goal"

    cd "$PROJECT_DIR"

    for session_num in $(seq 1 "$MAX_SESSIONS"); do

        # 检查是否已完成
        if check_done; then
            echo -e "${GREEN}✅ 所有任务已完成！${NC}"
            python3 -c "
import json
s = json.load(open('$STATE_FILE'))
s['status'] = 'completed'
json.dump(s, open('$STATE_FILE', 'w'), indent=2)
" 2>/dev/null || true
            break
        fi

        # 运行一轮
        run_session "$goal" "$session_num" || true

        # 检查完成状态
        if check_done; then
            echo -e "${GREEN}✅ 所有任务已完成！${NC}"
            break
        fi

        # 还有轮次，等待后继续
        if [ "$session_num" -lt "$MAX_SESSIONS" ]; then
            echo -e "${YELLOW}⏳ ${COOLDOWN}秒后启动下一轮...${NC}"
            echo -e "${YELLOW}   （按 Ctrl+C 可中断，进度已保存）${NC}"
            sleep "$COOLDOWN"
        fi
    done

    echo ""
    echo -e "${CYAN}═══ 执行摘要 ══════════════════════════════════════════${NC}"
    echo -e "  总 Session 数: ${session_num}"
    echo -e "  进度文件: ${PROGRESS_FILE}"
    echo -e "  日志目录: ${LOG_DIR}"
    if [ -f "$PROGRESS_FILE" ]; then
        echo ""
        echo -e "${CYAN}═══ 最终进度 ══════════════════════════════════════════${NC}"
        cat "$PROGRESS_FILE"
    fi
}

# ── 入口 ──────────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    echo "用法: $0 <目标描述> [项目目录] [最大轮次] [每轮max_turns] [冷却秒数]"
    echo ""
    echo "示例:"
    echo "  $0 \"构建用户认证模块\""
    echo "  $0 \"开发 Autopilot UI 看板\" ./kortex-frontend 10 200 15"
    echo ""
    echo "让 Claude Code 自己开发自己:"
    echo "  $0 \"构建 Claude Autopilot 可视化看板系统\" . 10"
    exit 1
fi

main "$@"
