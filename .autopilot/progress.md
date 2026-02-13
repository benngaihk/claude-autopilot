# Autopilot 进度记录

## 状态: 进行中

## 目标
用 antd-vue 来优化前端样式

## 已完成
（无）

## 进行中
- [ ] TASK-016: 引入 antd-vue CDN 依赖和 dayjs 插件
- [ ] TASK-017: 重写 Landing 页面使用 antd 组件 (Input, Textarea, Button, List)
- [ ] TASK-018: 重写顶栏和状态指示器使用 antd 组件 (Layout.Header, Tag, Badge, Space)
- [ ] TASK-019: 重写 Kanban 看板使用 antd Card 组件
- [ ] TASK-020: 重写 Live Output 和 Task Detail 面板 (Collapse, Drawer)
- [ ] TASK-021: 整体样式调优和暗色主题配置 (ConfigProvider darkAlgorithm)
- [ ] TASK-022: 运行自测并提交

## 待办
（无）

## 关键上下文
- 前端坚持单文件 CDN 方案，不用 npm 构建
- antd-vue 4.x 使用 CSS-in-JS，通过 CDN 引入 antd.min.js 即可
- 需要 dayjs + 多个 dayjs 插件作为依赖
- 暗色主题通过 antd ConfigProvider 的 theme.algorithm = antd.theme.darkAlgorithm 配置
- 端口: 8000
