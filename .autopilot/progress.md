# Autopilot 进度记录

## 状态: 进行中

## 目标
自我改进：为项目添加类似 OpenClaw 的记忆管理功能 + 无限上下文功能

## 已完成
- [x] TASK-001~006: 基础 Dashboard (Kanban/Progress/Graph/Control/Logs/Live Output)

## 进行中
- [ ] TASK-007: 实现记忆存储后端 — memory/ 目录管理、MEMORY.md 长期记忆 + YYYY-MM-DD.md 每日日志

## 待办
- [ ] TASK-008: 实现记忆管理 API — CRUD 接口 + 关键字搜索 + 日志追加（依赖 TASK-007）
- [ ] TASK-009: 实现上下文压缩前自动记忆刷新 — autopilot.sh 中检测上下文接近极限时触发记忆保存（依赖 TASK-007）
- [ ] TASK-010: 实现无限上下文功能 — 自动摘要 + 跨 session 上下文恢复 + 记忆注入 prompt（依赖 TASK-008, TASK-009）
- [ ] TASK-011: 实现 Memory Dashboard UI — 新增 Memory Tab 显示 MEMORY.md + 每日日志浏览器 + 记忆编辑器 + 搜索（依赖 TASK-008）
- [ ] TASK-012: 更新 autopilot.sh 集成记忆系统 — prompt 注入相关记忆 + session 结束自动保存记忆（依赖 TASK-009, TASK-010）
- [ ] TASK-013: 添加测试并运行 self-test 验证所有新 API（依赖 TASK-011）

## 关键上下文
- 技术栈: FastAPI (server.py) + Vue 3 CDN (index.html) + Bash (autopilot.sh)
- 服务端口: 8000
- 前端: 零构建 CDN 单文件
- 文件系统存储，无数据库
- 记忆系统设计参考 OpenClaw: 文件优先、Markdown 格式、两层记忆（长期 MEMORY.md + 短期每日日志）
- 无限上下文方案: 上下文压缩前自动刷新记忆 + 跨 session 记忆注入 + 自动摘要
