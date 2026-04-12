# 版本更新说明 - 2026-04-12

## 版本主题

新增 Prompt Playground 与模型调用观测能力。

## 本次后端更新内容

### 1. 新增模型调用日志能力

- 新增 `ModelCallLog` 表，用于记录模型调用来源、功能阶段、模型名、token、耗时、成本、成功状态、fallback 状态、错误信息
- 新增 `llm_runtime.py`，统一封装结构化模型调用与日志写入
- 新增 `llm_costs.py`，用于按模型估算 token 成本

### 2. 新增 Prompt 版本管理能力

- 新增 `PromptVersion` 表，用于保存 Playground 中的 prompt 历史版本
- 支持保存 `system_instruction`、`user_template`、模型参数与备注

### 3. 新增 Prompt Registry

- 新增 `prompt_registry.py`
- 统一注册可调试的 prompt 场景
- 首版支持：
  - `jd_keyword`
  - `follow_up_suggestions`
  - `evaluation_score`
  - `evaluation_summary`
  - `evaluation_items_highlights`
  - `evaluation_items_risks`
  - `interview_questions`
  - `interview_submit`

### 4. 新增观测接口

- `GET /observability/summary`
- `GET /observability/logs`
- `GET /observability/trends`

### 5. 新增 Prompt Playground 接口

- `GET /prompt-playground/scenarios`
- `GET /prompt-playground/versions`
- `POST /prompt-playground/versions`
- `POST /prompt-playground/run`

### 6. 改造现有模型调用链路

- `agent.py` 中的评估、总结、亮点、风险、追问建议等结构化调用已接入统一 runtime
- `interview_agent.py` 中的模拟面试出题与评分已接入统一 runtime

## 设计说明

- Prompt Playground 当前定位为内部开发调试工具
- 保存的 Prompt Version 不会自动替换正式生产 prompt
- 日志默认保存摘要与预览，不默认全量保存简历和 JD 原文
- 如果 provider 当前链路拿不到 token metadata，则允许 token / cost 为空，但仍记录耗时与成功状态

## 后续建议

1. 补齐流式聊天与流式评估的日志统计
2. 增加 Playground 双版本对比运行
3. 增加日志按模型 / prompt / feature 的聚合图表
4. 增加环境变量开关，控制 Playground 与 Observability 是否启用
