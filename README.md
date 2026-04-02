# Resume Agent (简历智能体)

## 🌟 项目简介
这是一个基于 Gemini 构建的招聘 AI 后端，覆盖 JD 分析、简历解析、流式评估、追问聊天和模拟面试智能体。与传统关键词匹配不同，它通过 RAG 检索和结构化输出，让 AI 能基于证据做招聘判断，并把结论映射回具体简历片段。

## 技术架构
1. 解析层：支持 `.docx` / `.pdf` 简历解析，利用自定义分段逻辑保留上下文。

2. 存储层：

- ChromaDB: 负责简历片段的向量化存储与相似度检索。
- PostgreSQL: 持久化存储用户对话历史。
- Redis: 提供高速会话缓存，优化响应延迟。

3. 推理层：调用 Gemini 模型，结合检索到的上下文生成专业评估、聊天回答与模拟面试结果。

## 核心功能
- 上传并解析简历，写入向量库
- 分析 JD 并提取关键词
- 流式返回简历评估过程与证据来源
- 对指定候选人发起流式追问，并返回回答依据
- 启动模拟面试智能体，生成 10 道问题并统一评分


## 快速开始
```bash
python server.py
```

启动后可在 Swagger 页面调试接口。

## 关键接口
- `POST /analyze_jd`
- `POST /ocr_jd_image`
- `POST /upload_resume`
- `GET /resumes`
- `POST /evaluate`
- `POST /evaluate_stream`
- `POST /chat`
- `POST /interview/start_stream`
- `POST /interview/submit`
- `POST /interview/history`
- `GET /interview/history/{session_id}`

## 学习笔记：AI 应用开发的思考
在开发本项目过程中，我重点解决以下 AI 工程化难题：

- 如何通过 N-results 调优 平衡上下文的完整性与 Token 成本。
- 如何设计 System Instruction 确保 AI 保持“职业经纪人”的客观冷峻，而非盲目乐观。
