# Autopilot 进度记录

## 状态: 已完成

## 已完成
- [x] TASK-001: 创建 requirements.txt 和项目基础结构 (requirements.txt)
- [x] TASK-002: 实现 FastAPI 后端 - REST API 读取任务数据 (server.py)
- [x] TASK-003: 实现 FastAPI 后端 - WebSocket 实时推送 (server.py)
- [x] TASK-004: 实现前端看板视图 - 四列看板 + 任务卡片 (index.html)
- [x] TASK-005: 实现任务详情面板 - commit diff、测试运行、依赖导航 (index.html, server.py)
- [x] TASK-006: 实现控制面板 - autopilot 启停、日志流、CLAUDE.md 编辑器 (index.html, server.py)
- [x] TASK-007: 实现依赖图 - Mermaid DAG + 关键路径高亮 (index.html)

## 关键上下文
- 后端: Python FastAPI + WebSocket, 单文件 server.py, lifespan handler
- 前端: 单文件 index.html, Vue 3 CDN + Tailwind CDN + marked.js + mermaid.js
- 数据源: ~/.claude/tasks/{session-uuid}/*.json
- 后端端口: 8000
- 启动命令: python3 server.py → http://localhost:8000
