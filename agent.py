import os
import sys
import tempfile
from chromadb import chromadb
import asyncio
import json
from google import genai
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from datetime import datetime
from database import AsyncSessionLocal
from resume_parser import ResumeParser
from models import ChatMessage, Resume, ResumeEvaluation, ResumeStatus


def get_project_tech_stack(project_name: str):
    """
    当用户询问某个项目的具体技术细节（如使用的中间件、数据库、框架）时，调用此函数获取详细信息。

    Args:
        project_name: 项目关键词，例如'架构设计'或'金融量化'
    """

    # 模拟一个从更深的数据库中检索更详细信息的过程
    detailed_db = {
        "架构设计": "使用微服务架构，主要技术栈包括Spring Boot, Docker, Kubernetes。",
        "金融量化": "使用Python进行量化策略开发，主要技术栈包括Pandas, NumPy, scikit-learn。",
    }

    return detailed_db.get(project_name, "该项目没有更详细的技术文档记录。")


def get_current_date(year: str):
    """
    当用户询问当前日期或时间相关的问题时，调用此函数获取当前日期。

    Args:
        year: 用户输入的年份关键词，例如'11年研发经验'中的'11年'，实际上当前日期可能不止11年了，所以需要获取当前年份来计算实际经验年限，简历中写了毕业年份2014年，所以可以通过当前年份减去2014年来计算实际经验年限。
    """

    return datetime.now().year - 2014  # 2014是简历中毕业的年份


class ResumeAgent:
    def __init__(self, redis_client=None):
        load_dotenv()
        self.client_ai = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )  # Initialize the Gemini API client

        self.model_name = os.getenv("GEMINI_MODEL_NAME")

        # 设定静态存储路径，使用ChromaDB来存储简历数据和对应的向量
        self.client_db = chromadb.PersistentClient(
            path="./chroma_db"
        )  # Initialize ChromaDB client
        self.collection = self.client_db.get_or_create_collection(
            name="my_resume"
        )  # Create or get the collection for trading data

        # 设定system instruction，明确AI的角色和行为准则
        self.system_instruction = """
        你是一个极其专业的职业经纪人
        请基于提供的简历片段回答问题。你要说真话，不要为了讨好用户而过度美化
        如果简历里没写，就直接说不知道，不要瞎编。
        """

        # 定义一个变量存储会话，初始为 None
        self.sessions = {}  # 结构 {"user:id": chat_session_object}

        # 加载持久化的历史纪录到内存
        self.histories = {}  # 结构 {"user:id": [messages...]}

        # 初始化redis连接
        self.redis_client = redis_client

        self.cache_expire = 3600  # 缓存1小时

        self.parser = ResumeParser(chunk_size=200, overlap=50)  # 初始化简历解析器

    async def extract_jd_info(self, jd_text: str):
        """
        直接调用AI抽取岗位、目标、约束，供function calling
        """
        prompt = f'请从以下JD描述中提取岗位名称、岗位目标、岗位约束，输出JSON：\nJD描述：{jd_text}\n输出示例：{{"role":"前端开发工程师","objective":"负责核心前端架构设计","constraints":"本科及以上学历，5年以上经验"}}'
        # 注意：此处需有event loop支持
        from google import genai
        import os

        client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model_name = os.getenv("GEMINI_MODEL_NAME")
        response = await client_ai.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        try:
            return json.loads(response.text)
        except Exception:
            return {
                "role": "岗位未知",
                "objective": "目标未知",
                "constraints": "无特殊约束",
            }

    async def analyze_jd_keywords(self, jd_text: str):
        """
        调用AI分析JD描述，返回关键词列表
        """
        prompt = f'请从以下JD描述中提取5个最重要的岗位关键词，输出JSON数组：\n{jd_text}\n输出示例：["Vue3","性能优化","团队管理"]'
        response = await self.client_ai.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        try:
            return json.loads(response.text)
        except Exception:
            return []

    async def check_and_get_evaluation(
        self, user_id: str, candidate_name: str, phone: str, db: AsyncSession = None
    ):
        # 查询数据库中是否存在该候选人的简历
        stmt = select(Resume).where(
            Resume.candidate_name == candidate_name,
            Resume.phone == phone,
            Resume.user_id == user_id,
        )

        result = await db.execute(stmt)
        record = result.scalar_one_or_none()
        return record

    async def evaluate_resume(
        self, resume_text: str, jd: str = None, jd_keywords: list = None
    ):
        """
        通过function calling获取岗位、目标、约束，不再写死prompt
        """
        # 1. 先让AI分析JD，抽取岗位、目标、约束
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_jd_info",
                    "description": "从JD描述中提取岗位、目标、约束",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "description": "岗位名称"},
                            "objective": {"type": "string", "description": "岗位目标"},
                            "constraints": {
                                "type": "string",
                                "description": "岗位约束",
                            },
                        },
                        "required": ["role", "objective", "constraints"],
                    },
                },
            }
        ]
        jd_info_prompt = (
            f"请从以下JD描述中提取岗位、目标、约束，调用extract_jd_info函数：\n{jd}"
        )
        jd_info_resp = await self.client_ai.aio.models.generate_content(
            model=self.model_name,
            contents=jd_info_prompt,
            tools=tools,
            config={"response_mime_type": "application/json"},
        )
        try:
            jd_info = json.loads(jd_info_resp.text)
        except Exception:
            jd_info = {"role": "", "objective": "", "constraints": ""}

        # 2. 生成评估prompt，动态拼接岗位、目标、约束
        prompt = f"""
        ROLE: {jd_info.get("role", "岗位未知")}
        OBJECTIVE: {jd_info.get("objective", "目标未知")}
        CONSTRAINTS: {jd_info.get("constraints", "无特殊约束")}
        
        JD描述: {jd}
        JD关键词: {", ".join(jd_keywords or [])}
        
        评估以下简历，判定是否符合岗位要求，并输出结构化评估结果。
        必须输出如下结构字段：
            - radar_scores: [技术深度, 项目经验, 软技能, 背景实力, AI提效]，每项0-100分
            - radar_indicators: [{"name": "技术深度", "max": 100}, ...]
            - tech_stack_citations: 每个技术栈的引用原文片段数组
            - key_achievements_citations: 每个成就的引用原文片段数组

        INPUT_RESUME:
        {resume_text}

        OUTPUT_FORMAT:
        
        请严格按照如下JSON格式输出：
        {{
            "decision": "评估结论",
            "match_score": 92,
            "decision_range": "Highly Frontend Expert",
            "radar_scores": [80, 90, 70, 85, 75],
            "radar_indicators": [
                {{"name": "技术深度", "max": 100}}, # 对比 JD 要求的框架深度（如 Vue3 原理、性能优化经验）
                {{"name": "项目经验", "max": 100}}, # 对比简历中的项目复杂度和JD 要求的项目经验
                {{"name": "软技能", "max": 100}}, # 对比 JD 要求的软技能
                {{"name": "背景实力", "max": 100}}, # 对比 JD 要求的背景实力
                {{"name": "AI提效", "max": 100}} # 对比 JD 要求的 AI 提效能力
            ],
            "tech_stack": ["Vue 2/3", "TypeScript", "Electron"],
            "tech_stack_citations": ["原文片段1", "原文片段2", "原文片段3"],
            "key_achievements": ["主导Electron桌面端重构", "JsonSchema元数据驱动"],
            "key_achievements_citations": ["原文片段A", "原文片段B"],
            "ai_bonus": "深度集成AI开发流",
            "risks": ["2022年及2025年两段工作经历时间较短"]
        }}
        """

        response = await self.client_ai.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )

        # 转换为python对象
        result = json.loads(response.text)
        # 兼容前端字段
        result["radarData"] = result.get("radar_scores")
        result["radarIndicators"] = result.get("radar_indicators")
        result["citations"] = (result.get("key_achievements_citations") or []) + (
            result.get("tech_stack_citations") or []
        )
        return result

    async def delete_old_vector_data(self, phone: str):
        """
        根据手机号从 ChromaDB 中删除旧数据
        """
        self.collection.delete(where={"phone": phone})

    async def extract_info_from_resume(self, resume_text: str):
        """
        使用 AI 从简历中提取关键标识信息
        """
        prompt = f"""
        请从以下简历内容中提取候选人的姓名和手机号。
        如果简历中没有提到手机号，请返回 "None"。
        
        简历内容：
        {resume_text}
        
        输出格式（JSON）：
        {{
            "candidate_name": "姓名",
            "phone": "手机号"
        }}
        """
        response = await self.client_ai.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return json.loads(response.text)

    async def _extract_candidate_name(self, question=str):
        """从用户问题中提取候选人姓名的简单方法，没有返回None"""
        prompt = f"""
        从以下问题中提取候选人姓名，如果没有明确的姓名，返回'None'：
        问题：{question}
        """
        response = await self.client_ai.aio.models.generate_content(
            model=self.model_name, contents=prompt
        )
        name = response.text.strip()
        return name if name != "None" else None

    async def _process_file_and_vector(
        self, file_content: bytes, candidate_name: str, phone: str
    ):
        """
        处理文件内容：解析文本，生成向量，存入ChromaDB
        返回解析后的纯文本全文
        """

        temp_path = ""
        raw_text = ""
        # 1. 使用临时文件处理二进制流
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            temp_file.write(file_content)
            temp_file.flush()  # 确保数据写入磁盘
            temp_path = temp_file.name
        try:
            # 2. 提取文本
            raw_text = self.parser.extract_from_docx(temp_path)
            chunks = self.parser.get_chunks(raw_text)
        finally:
            # 3. 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

        self.collection.delete(where={"phone": phone})  # 删除旧数据
        embeddings = [self._get_embedding(chunk) for chunk in chunks]
        metadatas = [
            {
                "candidate_name": candidate_name,
                "phone": phone,
                "source": "uploaded_resume",
            }
            for _ in chunks
        ]
        ids = [f"{phone}_{i}" for i in range(len(chunks))]

        self.collection.upsert(
            ids=ids,
            metadatas=metadatas,
            documents=chunks,
            embeddings=embeddings,
        )
        return raw_text

    async def _update_status(
        self,
        db: AsyncSession,
        resume_id: int,
        new_status: ResumeStatus,
        content: str = None,
        evaluation_result: dict = None,
    ):
        """更新简历处理状态的辅助函数"""
        stmt = update(Resume).where(Resume.id == resume_id).values(status=new_status)
        if content is not None:
            stmt = stmt.values(content=content)
        if evaluation_result is not None:
            stmt = stmt.values(evaluation_result=evaluation_result)

        stmt = stmt.values(updated_at=datetime.now())  # 更新状态时也更新updated_at字段

        await db.execute(stmt)
        await db.commit()
        print(f"简历ID {resume_id} 状态已更新为 {new_status.value}")

    async def handle_resume_process(
        self, resume_id: int, file_content: bytes, candidate_name: str, phone: str
    ):
        # 这里需要一个新的db session
        async with AsyncSessionLocal() as db:
            try:
                # 1. 更新数据库状态为解析中
                await self._update_status(db, resume_id, ResumeStatus.PARSING)

                raw_text = await self._process_file_and_vector(
                    file_content, candidate_name, phone
                )

                await self._update_status(
                    db, resume_id, ResumeStatus.EVALUATING, content=raw_text
                )

                evaluation_result = await self.evaluate_resume(raw_text)

                await self._update_status(
                    db,
                    resume_id,
                    ResumeStatus.COMPLETED,
                    evaluation_result=evaluation_result,
                )
            except Exception as e:
                print(f"更新状态失败: {e}")
                await self._update_status(db, resume_id, ResumeStatus.FAILED)

    async def add_resume(self, file_path, candidate_name, phone):
        # 1. 提取并切分
        doc_text = self.parser.extract_from_docx(file_path)
        chunks = self.parser.get_chunks(doc_text)

        self.collection.delete(where={"phone": phone})

        # 生成向量
        embeddings = [self._get_embedding(chunk) for chunk in chunks]

        # 2. 准备元数据,每一段都带上候选人姓名
        metadatas = [
            {"candidate_name": candidate_name, "phone": phone, "source": file_path}
            for _ in chunks
        ]
        ids = [f"{phone}_{i}" for i in range(len(chunks))]

        # 3. 存入ChromaDB
        self.collection.upsert(
            ids=ids,
            metadatas=metadatas,
            documents=chunks,
            embeddings=embeddings,
        )

        test_res = self.collection.get(limit=1)
        print(f"📦 数据库状态检查: {test_res['documents']}")

        return len(chunks)

    async def _get_cache_key(self, user_id: str):
        return f"chat_cache:{user_id}"

    async def _save_message(
        self, user_id: str, role: str, content: str, db: AsyncSession = None
    ):
        """双写逻辑：PG 永久存，Redis 更新缓存"""

        # 1. 写入PostgreSQL数据库
        new_message = ChatMessage(
            user_id=user_id,
            role=role,
            content=content,
        )
        db.add(new_message)

        # 2. 更新Redis缓存
        cache_key = await self._get_cache_key(user_id)
        await self.redis_client.delete(
            cache_key
        )  # 删除旧缓存，下一次查询会从PG加载最新数据

    async def _load_history(self, user_id, limit: int = 10, db: AsyncSession = None):
        """双级缓存查询 Redis->PG"""
        cache_key = await self._get_cache_key(user_id)

        # 1. 先查Redis缓存
        cached_data = await self.redis_client.get(cache_key)
        if cached_data:
            print(f"从Redis缓存中加载用户 {user_id} 的历史记录")
            return json.loads(cached_data)

        """ redis没命中，从PG加载最近的历史记录"""
        print(f"[PG]Redis未命中，从数据库中读取用户 {user_id} 的历史记录")

        # 查最近的N条，按时间降序排列
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.user_id == user_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        messages = result.scalars().all()
        # 转换成模型需要的格式，注意要逆序（从旧到新）
        history = [
            {"role": msg.role, "parts": [{"text": msg.content}]}
            for msg in reversed(messages)
        ]

        if history:
            # 加载到Redis缓存，设置过期时间
            await self.redis_client.setex(
                cache_key, self.cache_expire, json.dumps(history)
            )

        return history

    def _get_embedding(self, text: str):
        result = self.client_ai.models.embed_content(
            model=os.getenv("GEMINI_EMBEDDING_MODEL_NAME"), contents=text
        )
        return result.embeddings[0].values

    async def ask(
        self, question: str, user_id: str, search_filter=None, db: AsyncSession = None
    ):
        if not db:
            raise ValueError("Database session is required")
        # 1. 如果会话没有创建，尝试从磁盘恢复
        if user_id not in self.sessions:
            print(f"为用户 {user_id} 创建新会话")
            user_history = await self._load_history(user_id, db=db)
            self.histories[user_id] = user_history

            self.sessions[user_id] = self.client_ai.aio.chats.create(
                model=self.model_name,
                history=user_history,  # 关键：把历史喂给新会话
                config={
                    "system_instruction": self.system_instruction,
                    "tools": [get_project_tech_stack, get_current_date],
                },
            )

        # 2. 完整的RAG问答流程
        detected_name = (
            await self._extract_candidate_name(question)
            if not search_filter
            else search_filter.get("candidate_name")
        )
        enhanced_query = (
            f"候选人:{detected_name} {question}" if detected_name else question
        )

        query_vector = self._get_embedding(enhanced_query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=8,
        )

        context = (
            "\n".join(results["documents"][0])
            if results["documents"][0]
            else "(未找到相关简历描述)"
        )

        # 3. 构造增强prompt
        full_input = f"【参考背景】：{context}\n\n【用户问题】：{question}"

        # 4. 使用chat_session.send_message_stream获取流式响应，边生成边存储完整回答内容
        full_response_content = ""
        try:
            response_stream = await self.sessions[user_id].send_message_stream(
                full_input
            )
            async for chunk in response_stream:
                if chunk.text:
                    content = chunk.text
                    full_response_content += content
                    yield content
        except Exception as e:
            error_msg = f"\n[服务异常]: {str(e)}"
            yield error_msg
            full_response_content += error_msg

        # 5. 持久化：每次对话完，更新磁盘上的记忆
        self.histories[user_id].append({"role": "user", "parts": [{"text": question}]})
        self.histories[user_id].append(
            {"role": "model", "parts": [{"text": full_response_content}]}
        )

        # 6. 异步写入数据库
        await self._save_message(user_id, "user", question, db=db)
        await self._save_message(user_id, "model", full_response_content, db=db)


# 程序入口
async def main():
    # 初始化agent
    # agent = ResumeAgent()

    print("AI职业经纪人已启动，输入'exit'退出")

    # # 准备简历数据
    # resumes = [
    #     {
    #         "id": "exp_1",
    #         "text": "9年研发经验，熟悉架构设计，主导过多个大型项目的开发。",
    #     },
    #     {"id": "exp_2", "text": "擅长金融量化交易系统开发，熟悉A股交易机制"},
    # ]

    # # 同步数据
    # agent.sync_data(resumes)

    # while True:
    #     user_input = input("请输入问题：")
    #     if user_input.lower() == "exit":
    #         print("退出程序")
    #         break
    #     answer = await agent.ask(user_input, user_id)
    #     print(f"AI的回答是：{answer}")


if __name__ == "__main__":
    asyncio.run(main())
