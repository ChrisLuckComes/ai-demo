import asyncio
import json
import importlib
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator, cast

from dotenv import load_dotenv
import cv2  # type: ignore[import-not-found]
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    ChatMessage,
    ChatSuggestionsResponse,
    EvaluationItemsResult,
    EvaluationResult,
    EvaluationScoreResult,
    EvaluationSummaryResult,
    JDAnalysisResponse,
    JDKeywordExtractionResult,
    OCRResponse,
)
from agent_prompts import (
    SYSTEM_INSTRUCTION,
    build_chat_prompt,
    build_evaluation_prompt,
    build_evaluation_items_prompt,
    build_evaluation_score_prompt,
    build_evaluation_summary_prompt,
    build_follow_up_prompt,
    build_jd_keyword_prompt,
)
from agent_utils import (
    build_chat_fallback,
    build_radar_metrics,
    build_resume_sources,
    chunk_to_text,
    coerce_model,
    fallback_evaluation,
    fallback_keywords,
    format_context_docs,
    format_sources_for_prompt,
    normalize_evaluation_items,
    normalize_keyword_candidates,
    normalize_source_ids,
    normalize_sources,
    to_langchain_message,
    unique_strings,
)
from resume_parser import ResumeParser


class ResumeAgent:
    def __init__(self, redis_client=None):
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = gemini_api_key

        base_dir = Path(__file__).resolve().parent
        # LLM 负责推理与生成，embedding 负责把简历切片转成向量用于检索。
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
            temperature=0.2,
        )
        # 评估链路更强调稳定性，因此单独使用低温模型，减少同输入下的结果漂移。
        self.evaluation_llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
            temperature=0,
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "models/text-embedding-004"),
        )
        rapidocr_module = self._load_rapidocr_module()
        self.ocr_engine = rapidocr_module.RapidOCR() if rapidocr_module is not None else None
        # Chroma 持久化保存简历分片，后续聊天和评估都从这里做语义检索。
        self.vector_store = Chroma(
            collection_name="resume_agent",
            embedding_function=self.embeddings,
            persist_directory=str(base_dir / "chroma_db"),
        )
        self.redis_client = redis_client
        self.cache_expire = 3600
        self.parser = ResumeParser(chunk_size=500, overlap=80)
        self.system_instruction = SYSTEM_INSTRUCTION
        self.jd_keyword_timeout_seconds = float(os.getenv("JD_KEYWORD_TIMEOUT_SECONDS", "15"))
        self.jd_keyword_prompt = build_jd_keyword_prompt()
        self.jd_keyword_chain = self.jd_keyword_prompt | self.llm.with_structured_output(JDKeywordExtractionResult)

    async def extract_text_from_image(self, file_bytes: bytes, mime_type: str) -> OCRResponse:
        if self.ocr_engine is None:
            raise ValueError("未安装 rapidocr_onnxruntime，无法执行图片OCR")

        # 先把上传的二进制图片解码成 OpenCV 可处理的矩阵，再交给 OCR 引擎识别。
        image_buffer = np.frombuffer(file_bytes, dtype=np.uint8)
        decoded_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        if decoded_image is None:
            raise ValueError("图片解码失败，无法执行OCR")

        result, _ = self.ocr_engine(decoded_image)
        if not result:
            return OCRResponse(text="")

        text = "\n".join(item[1] for item in result if len(item) > 1 and item[1].strip())
        return OCRResponse(text=text.strip())

    async def generate_follow_up_suggestions(
        self,
        question: str,
        answer: str,
        candidate_name: str | None = None,
    ) -> list[str]:
        prompt = build_follow_up_prompt()
        try:
            structured_llm = self.llm.with_structured_output(ChatSuggestionsResponse)
            raw_result = await (prompt | structured_llm).ainvoke(
                {
                    "candidate_name": candidate_name or "未指定",
                    "question": question,
                    "answer": answer,
                }
            )
            result: ChatSuggestionsResponse = coerce_model(
                raw_result, ChatSuggestionsResponse
            )
            suggestions = unique_strings(result.suggestions)
            if suggestions:
                return suggestions[:3]
        except Exception:
            pass

        base_name = candidate_name or "候选人"
        return [
            f"{base_name} 在最近一段项目中的具体职责是什么？",
            f"{base_name} 在关键技术方案里承担了多大 ownership？",
            f"针对当前岗位要求，{base_name} 最大的风险点是什么？",
        ]

    async def analyze_jd(self, jd_text: str) -> JDAnalysisResponse:
        cleaned_text = jd_text.strip()
        if not cleaned_text:
            return JDAnalysisResponse(keywords=[])

        return JDAnalysisResponse(keywords=await self._extract_jd_keywords(cleaned_text))

    async def _extract_jd_keywords(self, text: str) -> list[str]:
        try:
            raw_result = await asyncio.wait_for(
                self.jd_keyword_chain.ainvoke({"text": text}),
                timeout=self.jd_keyword_timeout_seconds,
            )
            result: JDKeywordExtractionResult = coerce_model(
                raw_result, JDKeywordExtractionResult
            )
            keywords = normalize_keyword_candidates(result.keywords, text)
            if keywords:
                return keywords[:8]
        except Exception:
            pass

        return fallback_keywords(text)

    async def evaluate_resume(
        self,
        resume_text: str,
        jd_text: str,
        jd_keywords: list[str] | None = None,
    ) -> dict:
        keywords, sources = await self.prepare_evaluation_context(
            resume_text=resume_text,
            jd_keywords=jd_keywords,
        )
        return await self.generate_evaluation(
            resume_text=resume_text,
            jd_text=jd_text,
            keywords=keywords,
            sources=sources,
        )

    async def prepare_evaluation_context(
        self,
        resume_text: str,
        jd_keywords: list[str] | None = None,
    ) -> tuple[list[str], list[dict[str, str]]]:
        keywords = self.require_jd_keywords(jd_keywords)
        # 先把简历拆成可引用证据，后面要求模型只能引用这些 source_id。
        sources = self.build_evaluation_sources(resume_text)
        return keywords, sources

    def require_jd_keywords(self, jd_keywords: list[str] | None) -> list[str]:
        keywords = unique_strings(jd_keywords or [])
        if not keywords:
            raise ValueError("请先分析JD")
        return keywords

    def build_evaluation_sources(self, resume_text: str) -> list[dict[str, str]]:
        return build_resume_sources(resume_text)

    async def generate_evaluation(
        self,
        resume_text: str,
        jd_text: str,
        keywords: list[str],
        sources: list[dict[str, str]],
    ) -> dict:
        source_block = format_sources_for_prompt(sources)

        try:
            structured_llm = self.evaluation_llm.with_structured_output(EvaluationResult)
            prompt = build_evaluation_prompt(self.system_instruction)
            raw_result = await (prompt | structured_llm).ainvoke(
                {
                    "keywords": ", ".join(keywords),
                    "jd_text": jd_text,
                    "resume_text": resume_text,
                    "source_block": source_block,
                }
            )
            result: EvaluationResult = coerce_model(raw_result, EvaluationResult)
            payload = result.model_dump()
            # sources 以服务端切分结果为准，不信任模型回传的 sources，避免它把整份简历原样回显。
            payload["sources"] = normalize_sources(None, sources)
            payload["summary_source_ids"] = normalize_source_ids(
                payload.get("summary_source_ids"), payload["sources"]
            )
            payload["highlights"] = normalize_evaluation_items(
                payload.get("highlights"), payload["sources"]
            )
            payload["risks"] = normalize_evaluation_items(payload.get("risks"), payload["sources"])
            if not payload.get("radar_metrics"):
                payload["radar_metrics"] = [
                    metric.model_dump()
                    for metric in build_radar_metrics(payload["match_score"])
                ]
            return payload
        except Exception:
            # 结构化输出失败时降级到规则评估，保证接口仍能返回可用结果。
            return fallback_evaluation(resume_text, keywords, sources)

    async def evaluate_resume_in_steps(
        self,
        resume_text: str,
        jd_text: str,
        jd_keywords: list[str] | None = None,
    ) -> dict:
        keywords, sources = await self.prepare_evaluation_context(
            resume_text=resume_text,
            jd_keywords=jd_keywords,
        )
        score_result = await self.generate_evaluation_score(
            resume_text=resume_text,
            jd_text=jd_text,
            keywords=keywords,
        )
        summary_result = await self.generate_evaluation_summary(
            resume_text=resume_text,
            jd_text=jd_text,
            keywords=keywords,
            sources=sources,
            match_score=score_result["match_score"],
            decision=score_result["decision"],
        )
        highlights = await self.generate_evaluation_items(
            item_type="highlights",
            resume_text=resume_text,
            jd_text=jd_text,
            keywords=keywords,
            sources=sources,
            match_score=score_result["match_score"],
            decision=score_result["decision"],
            summary=summary_result["summary"],
        )
        risks = await self.generate_evaluation_items(
            item_type="risks",
            resume_text=resume_text,
            jd_text=jd_text,
            keywords=keywords,
            sources=sources,
            match_score=score_result["match_score"],
            decision=score_result["decision"],
            summary=summary_result["summary"],
        )
        normalized_sources = normalize_sources(None, sources)
        return {
            "title": score_result["title"],
            "decision": score_result["decision"],
            "match_score": score_result["match_score"],
            "radar_metrics": self.build_radar_payload(score_result["match_score"]),
            "summary": summary_result["summary"],
            "summary_source_ids": normalize_source_ids(summary_result["summary_source_ids"], normalized_sources),
            "highlights": normalize_evaluation_items(highlights["items"], normalized_sources),
            "risks": normalize_evaluation_items(risks["items"], normalized_sources),
            "sources": normalized_sources,
        }

    def build_radar_payload(self, match_score: int) -> list[dict]:
        return [metric.model_dump() for metric in build_radar_metrics(match_score)]

    async def generate_evaluation_score(
        self,
        resume_text: str,
        jd_text: str,
        keywords: list[str],
    ) -> dict:
        try:
            structured_llm = self.evaluation_llm.with_structured_output(EvaluationScoreResult)
            prompt = build_evaluation_score_prompt(self.system_instruction)
            raw_result = await (prompt | structured_llm).ainvoke(
                {
                    "keywords": ", ".join(keywords),
                    "jd_text": jd_text,
                    "resume_text": resume_text,
                }
            )
            result: EvaluationScoreResult = coerce_model(raw_result, EvaluationScoreResult)
            payload = result.model_dump()
            payload["match_score"] = max(0, min(100, int(payload["match_score"])))
            return payload
        except Exception:
            fallback = fallback_evaluation(resume_text, keywords, [])
            return {
                "title": fallback["title"],
                "decision": fallback["decision"],
                "match_score": fallback["match_score"],
            }

    async def generate_evaluation_summary(
        self,
        resume_text: str,
        jd_text: str,
        keywords: list[str],
        sources: list[dict[str, str]],
        match_score: int,
        decision: str,
    ) -> dict:
        source_block = format_sources_for_prompt(sources)
        normalized_sources = normalize_sources(None, sources)
        try:
            structured_llm = self.evaluation_llm.with_structured_output(EvaluationSummaryResult)
            prompt = build_evaluation_summary_prompt(self.system_instruction)
            raw_result = await (prompt | structured_llm).ainvoke(
                {
                    "match_score": match_score,
                    "decision": decision,
                    "keywords": ", ".join(keywords),
                    "jd_text": jd_text,
                    "resume_text": resume_text,
                    "source_block": source_block,
                }
            )
            result: EvaluationSummaryResult = coerce_model(raw_result, EvaluationSummaryResult)
            payload = result.model_dump()
            payload["summary_source_ids"] = normalize_source_ids(payload.get("summary_source_ids"), normalized_sources)
            return payload
        except Exception:
            fallback = fallback_evaluation(resume_text, keywords, sources)
            return {
                "summary": fallback["summary"],
                "summary_source_ids": fallback["summary_source_ids"],
            }

    async def generate_evaluation_items(
        self,
        item_type: str,
        resume_text: str,
        jd_text: str,
        keywords: list[str],
        sources: list[dict[str, str]],
        match_score: int,
        decision: str,
        summary: str,
    ) -> dict:
        source_block = format_sources_for_prompt(sources)
        normalized_sources = normalize_sources(None, sources)
        try:
            structured_llm = self.evaluation_llm.with_structured_output(EvaluationItemsResult)
            prompt = build_evaluation_items_prompt(self.system_instruction, item_type)
            raw_result = await (prompt | structured_llm).ainvoke(
                {
                    "match_score": match_score,
                    "decision": decision,
                    "summary": summary,
                    "keywords": ", ".join(keywords),
                    "jd_text": jd_text,
                    "resume_text": resume_text,
                    "source_block": source_block,
                }
            )
            result: EvaluationItemsResult = coerce_model(raw_result, EvaluationItemsResult)
            payload = result.model_dump()
            payload["items"] = normalize_evaluation_items(payload.get("items"), normalized_sources)
            return payload
        except Exception:
            fallback = fallback_evaluation(resume_text, keywords, sources)
            fallback_key = "highlights" if item_type == "highlights" else "risks"
            return {"items": fallback[fallback_key]}

    def build_provisional_evaluation(
        self,
        resume_text: str,
        keywords: list[str],
        sources: list[dict[str, str]],
    ) -> dict:
        return fallback_evaluation(resume_text, keywords, sources)

    async def ingest_resume(
        self,
        file_name: str,
        file_content: bytes,
        user_id: str,
        resume_id: int,
        candidate_name: str,
        phone: str,
    ) -> str:
        suffix = Path(file_name).suffix.lower() or ".docx"
        if suffix not in {".docx", ".pdf"}:
            raise ValueError("仅支持 .docx 或 .pdf 格式的简历文件")

        temp_path = ""
        try:
            # 解析库依赖本地文件路径，因此先把上传文件落到临时文件。
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            raw_text = self.parser.extract_text(temp_path)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        if not raw_text.strip():
            raise ValueError("简历内容为空，无法继续处理")

        # 每份简历会被切成多个 chunk 入向量库，便于后续按问题检索局部证据。
        chunks = self.parser.get_chunks(raw_text)
        self.delete_resume_vectors(user_id=user_id, resume_id=resume_id)
        self.vector_store.add_texts(
            texts=chunks,
            metadatas=[
                {
                    "user_id": user_id,
                    "resume_id": str(resume_id),
                    "candidate_name": candidate_name,
                    "phone": phone,
                }
                for _ in chunks
            ],
            ids=[f"{user_id}:{resume_id}:{index}" for index in range(len(chunks))],
        )
        return raw_text

    def delete_resume_vectors(
        self,
        user_id: str,
        resume_id: int | None = None,
        candidate_name: str | None = None,
    ) -> None:
        where = self._build_vector_filter(
            user_id=user_id, resume_id=resume_id, candidate_name=candidate_name
        )
        if where:
            self.vector_store.delete(where=where)

    async def ask(
        self,
        question: str,
        user_id: str,
        db: AsyncSession,
        candidate_name: str | None = None,
        resume_id: int | None = None,
    ) -> str:
        chunks: list[str] = []
        async for chunk in self.stream_chat(
            question=question,
            user_id=user_id,
            db=db,
            candidate_name=candidate_name,
            resume_id=resume_id,
        ):
            chunks.append(chunk)
        # 非流式接口本质上复用了流式实现，只是把分块结果重新拼接起来。
        return "".join(chunks)

    async def get_chat_sources(
        self,
        question: str,
        user_id: str,
        candidate_name: str | None = None,
        resume_id: int | None = None,
    ) -> list[dict[str, str]]:
        return await self._retrieve_context_docs(
            question=question,
            user_id=user_id,
            candidate_name=candidate_name,
            resume_id=resume_id,
        )

    async def stream_chat(
        self,
        question: str,
        user_id: str,
        db: AsyncSession,
        candidate_name: str | None = None,
        resume_id: int | None = None,
    ) -> AsyncIterator[str]:
        # 先读历史消息，再检索当前问题最相关的简历片段，最后把二者一起交给 LLM。
        chat_history = await self._load_history(user_id=user_id, db=db)
        context_docs = await self._retrieve_context_docs(
            question=question,
            user_id=user_id,
            candidate_name=candidate_name,
            resume_id=resume_id,
        )
        context = format_context_docs(context_docs)

        prompt = build_chat_prompt(self.system_instruction)
        chain = prompt | self.llm

        response_parts: list[str] = []
        try:
            async for chunk in chain.astream(
                {
                    "chat_history": chat_history,
                    "context": context or "未检索到匹配的简历片段。",
                    "question": question,
                }
            ):
                text = chunk_to_text(chunk)
                if text:
                    response_parts.append(text)
                    yield text
        except Exception:
            # 流式生成失败时仍返回基于检索结果的降级答案，避免前端整段报错。
            fallback = build_chat_fallback(question=question, context=context)
            response_parts.append(fallback)
            yield fallback

        full_response = "".join(response_parts).strip()
        # 用户问题和 AI 最终回答都会落库，用于多轮对话与缓存重建。
        await self._save_message(user_id=user_id, role="user", content=question, db=db)
        await self._save_message(user_id=user_id, role="ai", content=full_response, db=db)

    async def _retrieve_context_docs(
        self,
        question: str,
        user_id: str,
        candidate_name: str | None,
        resume_id: int | None,
    ) -> list[dict[str, str]]:
        search_kwargs: dict[str, object] = {"k": 6}
        # 通过 metadata filter 把检索范围限制到当前用户/当前简历，避免串数据。
        where = self._build_vector_filter(
            user_id=user_id, resume_id=resume_id, candidate_name=candidate_name
        )
        if where:
            search_kwargs["filter"] = where

        try:
            docs = await self.vector_store.as_retriever(search_kwargs=search_kwargs).ainvoke(
                question
            )
        except Exception:
            return []

        context_docs: list[dict[str, str]] = []
        # 给每个召回片段分配一个 source_id，方便前端展示“答案依据”。
        for index, doc in enumerate(docs, start=1):
            snippet = doc.page_content.strip()
            if not snippet:
                continue
            context_docs.append(
                {
                    "source_id": f"ctx_{index}",
                    "snippet": snippet[:700],
                }
            )
        return context_docs

    async def _save_message(
        self, user_id: str, role: str, content: str, db: AsyncSession
    ) -> None:
        db.add(ChatMessage(user_id=user_id, role=role, content=content))
        await db.flush()
        await self._cache_delete(await self._get_cache_key(user_id))

    async def _load_history(self, user_id: str, db: AsyncSession) -> list[BaseMessage]:
        cache_key = await self._get_cache_key(user_id)
        cached = await self._cache_get(cache_key)
        if cached:
            return self._history_from_cache(cached)

        # 历史消息优先从 Redis 取，缓存未命中时再回源数据库。
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.user_id == user_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(12)
        )
        result = await db.execute(stmt)
        messages = cast(list[ChatMessage], result.scalars().all())
        await self._cache_set(
            cache_key,
            json.dumps(
                [{"role": message.role, "content": message.content} for message in messages],
                ensure_ascii=False,
            ),
        )
        return [to_langchain_message(message.role, message.content) for message in messages]

    async def _get_cache_key(self, user_id: str) -> str:
        return f"chat_cache:{user_id}"

    async def _cache_get(self, key: str) -> str | None:
        if not self.redis_client:
            return None
        return await self.redis_client.get(key)

    async def _cache_set(self, key: str, value: str) -> None:
        if self.redis_client:
            await self.redis_client.setex(key, self.cache_expire, value)

    async def _cache_delete(self, key: str) -> None:
        if self.redis_client:
            await self.redis_client.delete(key)

    def _history_from_cache(self, cached: str) -> list[BaseMessage]:
        try:
            payload = json.loads(cached)
        except json.JSONDecodeError:
            return []

        history: list[BaseMessage] = []
        for item in payload:
            role = str(item.get("role", ""))
            content = str(item.get("content", ""))
            if content:
                history.append(to_langchain_message(role, content))
        return history

    def _load_rapidocr_module(self):
        try:
            return importlib.import_module("rapidocr_onnxruntime")
        except ImportError:
            return None

    def _build_vector_filter(
        self,
        user_id: str,
        resume_id: int | None = None,
        candidate_name: str | None = None,
    ) -> dict | None:
        clauses = [{"user_id": user_id}]
        if resume_id is not None:
            clauses.append({"resume_id": str(resume_id)})
        if candidate_name:
            clauses.append({"candidate_name": candidate_name})

        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
