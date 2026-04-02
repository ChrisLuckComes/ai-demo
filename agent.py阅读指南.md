# agent.py 阅读指南

这份文档不是解释每一行代码，而是帮你抓住 `agent.py` 的主流程。读懂下面这条链路之后，再回头看源码会轻松很多。

## 一句话理解

`ResumeAgent` 做的事情可以概括成 4 步：

1. 解析输入资料
2. 把简历存成可检索的向量片段
3. 根据问题或 JD 检索相关证据
4. 调用大模型生成结构化结果或聊天回答

---

## 1. 初始化阶段

入口：`D:\code\resume-agent\agent.py:38`

`__init__` 里主要初始化了 5 个核心能力：

- `self.llm`
  - 负责所有生成类任务
  - 比如 JD 关键词提取、简历评估、追问建议、聊天回答

- `self.embeddings`
  - 负责把文本转成向量
  - 给 Chroma 做语义检索用

- `self.ocr_engine`
  - 负责图片 OCR
  - 用来从 JD 截图里提文字

- `self.vector_store`
  - 简历切片的向量存储
  - 聊天和评估时的证据都从这里查

- `self.parser`
  - 负责解析 `.docx/.pdf`
  - 并把简历切成 chunk

另外还有一个很重要的变量：

- `self.system_instruction`
  - 这是全局 prompt 规则
  - 作用是把模型约束成“专业、客观、重证据的招聘顾问”

---

## 2. 上传简历后发生了什么

主入口方法：`D:\code\resume-agent\agent.py:266`

方法名：`ingest_resume`

这是“简历入库链路”的核心。

### 执行顺序

1. 校验文件类型，只允许 `.docx/.pdf`
2. 把上传文件写到临时文件
3. 调 `self.parser.extract_text()` 提取纯文本
4. 校验文本不能为空
5. 调 `self.parser.get_chunks()` 把文本切片
6. 先删掉旧向量，避免重复数据
7. 把 chunk + metadata 写入 Chroma

### 你要重点理解的点

- 为什么要切 chunk
  - 因为后面用户问问题时，不是整份简历都塞给模型
  - 而是先从 chunk 里检索最相关的几段

- metadata 有什么用
  - 这里保存了 `user_id`、`resume_id`、`candidate_name`、`phone`
  - 后面检索时就能限制“只查当前用户、当前候选人”

---

## 3. JD 分析链路

入口：`D:\code\resume-agent\agent.py:133`

核心方法：

- `analyze_jd`
- `_extract_jd_keywords`

### 它做了什么

把一段 JD 文本交给模型，抽取出真正适合招聘筛选的关键词。

### 输出结果是什么

- 例如：`React`、`TypeScript`、`Node.js`、`BFF`、`中后台` 这类词

### 为什么要单独做这一步

因为后面评估简历时：

- 需要知道 JD 的重点能力是什么
- 雷达图也需要维度来源
- 聊天追问也会参考 JD 重点

### 降级逻辑

如果结构化输出失败，会走 `agent_utils.fallback_keywords`

这个方法不会调用模型，而是：

- 从预置技术词库里匹配
- 再用正则从文本里提词
- 最后去重

也就是说：

- 正常情况：走 LLM
- 异常情况：走规则兜底

---

## 4. 简历评估链路

入口：`D:\code\resume-agent\agent.py`

核心方法：`evaluate_resume`

这是整个项目最关键的方法之一。

### 执行顺序

1. 先抽 JD 关键词
2. 生成 `sources`
3. 把 `sources` 格式化成 prompt 可引用证据
4. 调用结构化输出模型，要求返回：
   - summary
   - title
   - decision
   - match_score
   - radar_metrics
   - highlights
   - risks
   - sources
5. 对模型结果做清洗和收口
6. 如果失败，走 `agent_utils.fallback_evaluation`

### 这里最值得你重点理解的是 `sources`

相关方法：

- `prepare_evaluation_context`
- `generate_evaluation`
- `agent_utils.py` 中的 `build_resume_sources`
- `agent_utils.py` 中的 `format_sources_for_prompt`
- `agent_utils.py` 中的 `normalize_sources`
- `agent_utils.py` 中的 `normalize_source_ids`
- `agent_utils.py` 中的 `normalize_evaluation_items`

这套设计的意义是：

- 先把简历拆成证据片段
- 给每个片段一个 `source_id`
- 要求模型输出结论时只能引用这些 `source_id`
- 再由后端验证 `source_id` 是否真的合法

这就是你现在项目里“可解释性”的核心实现。

### 为什么还要后处理清洗

因为模型即使被要求结构化输出，也可能：

- 漏字段
- source_id 写错
- 返回字符串而不是对象
- sources 重复

所以必须在后端再做一次收口。

---

## 5. 聊天链路

入口：`D:\code\resume-agent\agent.py`

核心方法：`stream_chat`

### 执行顺序

1. 先查历史消息 `_load_history`
2. 再检索当前问题相关的简历片段 `_retrieve_context_docs`
3. 把“历史 + 检索结果 + 用户问题”一起拼进 prompt
4. 用 `chain.astream()` 流式生成回答
5. 每拿到一个 chunk 就 `yield`
6. 最后把完整回答写入数据库

### 相关辅助方法

- `ask`
  - 非流式接口的封装
  - 本质上复用 `stream_chat`

- `get_chat_sources`
  - 单独给前端返回这次聊天检索到的来源片段

- `_retrieve_context_docs`
  - 真正负责从 Chroma 召回相关证据

### `_retrieve_context_docs` 的关键点

位置：`D:\code\resume-agent\agent.py` 中的 `_retrieve_context_docs`

它做了两件重要的事：

1. 用 metadata filter 限定检索范围
   - 防止查到别人的简历

2. 给每个召回片段分配 `ctx_1`、`ctx_2` 这种 `source_id`
   - 前端就可以展示“这条回答依据来自哪些片段”

---

## 6. 历史消息与缓存

入口：`D:\code\resume-agent\agent.py` 中的 `_load_history`

核心方法：`_load_history`

### 逻辑

1. 先查 Redis
2. 命中就直接反序列化返回
3. 未命中再查数据库
4. 把数据库结果重新写回 Redis

### 为什么要缓存

因为聊天场景里每轮都要带历史消息：

- 不缓存会频繁查库
- 缓存可以减少延迟

相关方法：

- `_cache_get`
- `_cache_set`
- `_cache_delete`
- `_history_from_cache`

---

## 7. 降级逻辑

你这个项目有一个很好的工程点：不是一出错就崩，而是尽量降级。

### 关键词提取降级

方法：`agent_utils.fallback_keywords`

模型失败时：

- 用技术词库 + 正则提词

### 评估降级

方法：`agent_utils.fallback_evaluation`

模型失败时：

- 根据 JD 关键词在简历中的覆盖率粗略打分
- 生成 summary / highlights / risks
- 仍然带上 sources

### 聊天降级

方法：`agent_utils.build_chat_fallback`

流式调用失败时：

- 直接把已检索到的简历片段裁一段返回
- 告诉用户“这是降级回答”

这说明你的项目已经有 AI 工程化意识，而不是纯 demo。

---

## 8. 模拟面试智能体链路

如果你继续看“模拟面试”这条线，建议把它理解成 `ResumeAgent` 旁边新增的一条子流程。

主入口文件：`D:\code\resume-agent\interview_agent.py`

核心职责：

1. 复用 `ResumeAgent` 已经准备好的 JD / 简历 / 评估上下文
2. 生成 10 道结构化模拟面试题
3. 接收整场回答并做统一评分
4. 用缓存和 fallback 保证体验稳定

### 题目生成链路

核心方法：

- `_prepare_interview_materials`
- `prepare`
- `generate_interview_questions`

执行顺序：

1. 先校验简历内容是否存在
2. 复用 `agent.prepare_evaluation_context()` 准备关键词和证据来源
3. 如果数据库里已有评估结果，就直接复用；没有再补一次评估
4. 调面试 prompt 生成结构化题目
5. 清洗 `source_ids`
6. 写入 Redis 缓存

### 评分链路

核心方法：

- `submit`
- `submit_interview_answers`

执行顺序：

1. 同样先准备简历、JD、评估上下文
2. 把 10 道题和答案拼成 `qa_block`
3. 让模型返回逐题评分和整场反馈
4. 后端再根据平均分强制计算 `verdict`
5. 如果模型失败，走长度规则 fallback

这条链路的价值在于：

- 它复用了主评估能力
- 但没有继续挤进 `agent.py`
- 所以主评估 agent 和面试 agent 的边界更清楚

---

## 9. 你读源码时建议的顺序

我建议你按这个顺序看 `agent.py`：

1. `__init__`
2. `ingest_resume`
3. `_extract_jd_keywords`
4. `evaluate_resume`
5. `prepare_evaluation_context` / `generate_evaluation` / `agent_utils.normalize_*`
6. `stream_chat`
7. `_retrieve_context_docs`
8. `_load_history`
9. `interview_agent.py`
10. 各种 fallback 方法

这样你会先理解主流程，再理解细节工具函数。

---

## 10. 你现在要能口头讲出来的内容

如果面试官问你 `agent.py` 怎么工作的，你至少可以这样回答：

“我的 `ResumeAgent` 本质上是一个 RAG 驱动的招聘顾问 agent。上传简历后，我会先解析文档并切成 chunk，写入 Chroma 向量库。评估简历或聊天时，会先根据问题或 JD 检索相关片段，再把检索结果和 prompt 约束一起交给大模型生成结构化输出。为了提高可解释性，我额外设计了 source_id 机制，让 summary、亮点、风险和聊天回答都能映射回具体证据片段。为了保证稳定性，我还做了缓存、结构化输出清洗和 fallback 降级逻辑。”

这段如果你能顺畅讲出来，说明你已经真正理解这份代码了。
