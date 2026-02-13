# 自举启动：让 Claude Code 自己开发 Autopilot

## 快速开始

### 1. 准备工作

```bash
# 确保 Claude Code 已安装
claude --version

# 创建项目目录
mkdir -p claude-autopilot
cd claude-autopilot
git init

# 复制文件进来
cp /path/to/autopilot.sh .
cp /path/to/CLAUDE.md .
chmod +x autopilot.sh
```

### 2. 启动自举（让 Claude Code 自己开发 Autopilot）

```bash
./autopilot.sh "构建 Claude Autopilot 可视化管理系统。

## 目标产物

一个 Web UI + Python 后端，用来可视化管理 Claude Code Agent Teams 的任务执行。

## 技术栈
- 后端: Python FastAPI + WebSocket
- 前端: 单文件 HTML（内嵌 Vue 3 CDN + Tailwind CDN，不要用 npm 构建）
- 数据源: 直接读取 ~/.claude/tasks/ 和 ~/.claude/teams/ 目录

## 核心功能（按优先级）

### P0: 看板视图
- 四列看板: pending / in_progress / completed / failed
- 读取 ~/.claude/tasks/{team}/*.json 渲染任务卡片
- 卡片显示: ID、标题、owner、状态、依赖关系
- WebSocket 实时刷新（watchfiles 监听文件变化）
- 支持选择不同的 team

### P1: 任务详情面板
- 点击卡片右侧滑出详情
- 📋 方案 Tab: 显示 .autopilot/plans/TASK-ID.md（Markdown 渲染）
- 💻 代码 Tab: 执行 git log --grep=[TASK-ID] 获取关联 commit，展示文件列表和 diff
- 🧪 测试 Tab: 一个按钮触发 pytest / playwright test，流式显示输出
- 💬 通信 Tab: 读取 teammate inbox JSON 展示 Agent 间消息

### P2: 控制面板
- 启动/停止 autopilot.sh 的按钮
- 实时日志流（读 .autopilot/logs/）
- 规则配置编辑器（编辑 CLAUDE.md）

### P3: 依赖图
- 用 Mermaid.js 渲染任务 DAG
- 显示关键路径

## 文件结构
claude-autopilot/
├── autopilot.sh        # 已有 — 外层循环
├── CLAUDE.md           # 已有 — 工作指南
├── server.py           # FastAPI 后端（单文件）
├── index.html          # 前端（单文件，CDN 引入）
├── requirements.txt    # Python 依赖
└── .autopilot/
    ├── progress.md
    └── logs/

## 约束
- 前端必须是单个 HTML 文件，用 CDN 引入 Vue 3 和 Tailwind，不要用 npm
- 后端必须是单个 Python 文件
- 先做 P0 能跑起来，再做 P1、P2、P3
- 每完成一个优先级就 git commit
"
```

### 3. 观察执行

```bash
# 另一个终端查看进度
watch -n 5 cat .autopilot/progress.md

# 查看日志
tail -f .autopilot/logs/session_001_*.log

# 查看 git 提交
git log --oneline

# 如果中途想停
# Ctrl+C 停止当前 session，进度已保存
# 再次运行 ./autopilot.sh "同样的目标" 会从断点继续
```

### 4. 启动 UI（Claude Code 开发完成后）

```bash
pip install -r requirements.txt
python server.py
# 打开 http://localhost:8000
```

## 它会怎么跑

```
Session 1:
  → 分析需求
  → 拆解为 ~8 个任务
  → 写 progress.md
  → 开始 P0: 创建 server.py 基础框架
  → 创建 index.html 看板骨架
  → git commit
  → 更新 progress.md
  → (如果 context 还够就继续 P0)

Session 2:
  → 读 progress.md，了解上次做到哪
  → 继续 P0: WebSocket 实时刷新
  → 完成 P0，git commit
  → 开始 P1: 任务详情面板
  → ...

Session 3-N:
  → 继续未完成的工作
  → 每完成一个 P 级别就 commit
  → 全部完成 → 状态改为"已完成" → 外层脚本停止
```

## 验证可行性的检查清单

跑完后检查这些：

- [ ] `.autopilot/progress.md` 有内容且格式正确
- [ ] git log 有多个按任务分的 commit
- [ ] `server.py` 存在且可以 `python server.py` 启动
- [ ] `index.html` 存在且浏览器能打开
- [ ] 如果中间 Ctrl+C 重启，能从断点继续（最关键的验证点）

## 问题排查

**Claude Code 报错退出**
→ 正常，外层脚本会自动重启下一轮

**progress.md 没更新**
→ 可能 Claude Code 在 context 耗尽前没来得及写。
→ 手动在 progress.md 里写当前进度，重新启动

**Agent Teams 不工作**
→ 检查: `claude settings get env.CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`
→ 如果不是 Agent Teams 的场景（任务简单），Claude Code 会自己选择不用，这是正常的

**Session 一直重启不结束**
→ 检查 progress.md 里的"状态"字段是否被正确更新
→ 可以手动改为 `## 状态: 已完成` 来停止
