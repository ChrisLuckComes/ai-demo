import hashlib
import json

from agent import ResumeAgent
from agent_prompts import build_interview_questions_prompt, build_interview_submit_prompt
from agent_utils import (
    coerce_model,
    format_sources_for_prompt,
    normalize_evaluation_items,
    normalize_sources,
    unique_strings,
)
from models import (
    InterviewAnswerInput,
    InterviewEvaluationLLMResult,
    InterviewQuestionResult,
    InterviewQuestionsResponse,
    InterviewSubmitResult,
    Resume,
)


class InterviewAgent:
    def __init__(self, agent: ResumeAgent, redis_client=None, cache_ttl: int = 60 * 60 * 6):
        self.agent = agent
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl

    def build_cache_key(self, user_id: str, resume: Resume, jd_text: str) -> str:
        jd_hash = hashlib.md5(jd_text.encode("utf-8")).hexdigest()
        resume_version = int(resume.updated_at.timestamp()) if resume.updated_at else 0
        return f"interview:{user_id}:{resume.id}:{jd_hash}:{resume_version}"

    async def _prepare_interview_materials(
        self,
        *,
        resume: Resume,
        jd_text: str,
        jd_keywords: list[str] | None,
    ) -> tuple[list[str], dict, list[dict[str, str]]]:
        if not resume.content:
            raise ValueError("resume content missing")

        keywords, sources = await self.agent.prepare_evaluation_context(
            resume_text=resume.content,
            jd_keywords=jd_keywords,
        )
        evaluation = resume.evaluation_result or await self.agent.evaluate_resume(
            resume_text=resume.content,
            jd_text=jd_text,
            jd_keywords=jd_keywords,
        )
        return keywords, evaluation, sources

    async def get_cached_questions(self, cache_key: str) -> list[dict] | None:
        if not self.redis_client:
            return None
        cached = await self.redis_client.get(cache_key)
        if not cached:
            return None
        try:
            payload = json.loads(cached)
        except json.JSONDecodeError:
            return None
        questions = payload.get("questions")
        return questions if isinstance(questions, list) else None

    async def set_cached_questions(self, cache_key: str, questions: list[dict]) -> None:
        if not self.redis_client:
            return
        await self.redis_client.setex(
            cache_key,
            self.cache_ttl,
            json.dumps({"questions": questions}, ensure_ascii=False),
        )

    async def prepare(
        self,
        *,
        user_id: str,
        resume: Resume,
        jd_text: str,
        jd_keywords: list[str] | None,
    ) -> list[dict]:
        cache_key = self.build_cache_key(user_id, resume, jd_text)
        cached_questions = await self.get_cached_questions(cache_key)
        if cached_questions:
            return cached_questions

        keywords, evaluation, sources = await self._prepare_interview_materials(
            resume=resume,
            jd_text=jd_text,
            jd_keywords=jd_keywords,
        )
        questions = await self.generate_interview_questions(
            candidate_name=resume.candidate_name or "候选人",
            resume_text=resume.content or "",
            jd_text=jd_text,
            keywords=keywords,
            evaluation=evaluation,
            sources=sources,
        )
        await self.set_cached_questions(cache_key, questions)
        return questions

    async def submit(
        self,
        *,
        resume: Resume,
        jd_text: str,
        jd_keywords: list[str] | None,
        answers: list[InterviewAnswerInput],
    ) -> dict:
        keywords, evaluation, _ = await self._prepare_interview_materials(
            resume=resume,
            jd_text=jd_text,
            jd_keywords=jd_keywords,
        )
        return await self.submit_interview_answers(
            candidate_name=resume.candidate_name or "候选人",
            resume_text=resume.content or "",
            jd_text=jd_text,
            keywords=keywords,
            evaluation=evaluation,
            answers=answers,
        )

    async def warm(
        self,
        *,
        user_id: str,
        resume: Resume,
        jd_text: str,
        jd_keywords: list[str] | None,
    ) -> None:
        try:
            await self.prepare(
                user_id=user_id,
                resume=resume,
                jd_text=jd_text,
                jd_keywords=jd_keywords,
            )
        except Exception:
            return

    async def generate_interview_questions(
        self,
        candidate_name: str,
        resume_text: str,
        jd_text: str,
        keywords: list[str],
        evaluation: dict | None,
        sources: list[dict[str, str]],
    ) -> list[dict]:
        source_block = format_sources_for_prompt(sources)
        normalized_sources = normalize_sources(None, sources)
        evaluation_payload = evaluation or {}
        highlights = normalize_evaluation_items(evaluation_payload.get("highlights"), normalized_sources)
        risks = normalize_evaluation_items(evaluation_payload.get("risks"), normalized_sources)
        match_score = int(evaluation_payload.get("match_score") or 0)
        decision = str(evaluation_payload.get("decision") or "待进一步评估")
        highlights_text = "\n".join(f"- {item['text']}" for item in highlights) or "- 暂无明确亮点"
        risks_text = "\n".join(f"- {item['text']}" for item in risks) or "- 暂无明确风险"

        try:
            structured_llm = self.agent.evaluation_llm.with_structured_output(InterviewQuestionsResponse)
            prompt = build_interview_questions_prompt(self.agent.system_instruction)
            raw_result = await (prompt | structured_llm).ainvoke(
                {
                    "candidate_name": candidate_name,
                    "match_score": match_score,
                    "decision": decision,
                    "keywords": ", ".join(keywords),
                    "jd_text": jd_text,
                    "resume_text": resume_text,
                    "source_block": source_block,
                    "highlights_text": highlights_text,
                    "risks_text": risks_text,
                }
            )
            result: InterviewQuestionsResponse = coerce_model(raw_result, InterviewQuestionsResponse)
            questions = []
            allowed_source_ids = {source["source_id"] for source in normalized_sources}
            for index, item in enumerate(result.questions[:10], start=1):
                questions.append(
                    {
                        "question_id": item.question_id or f"q{index}",
                        "category": item.category,
                        "question": item.question,
                        "intent": item.intent,
                        "source_ids": [
                            source_id for source_id in item.source_ids if source_id in allowed_source_ids
                        ],
                    }
                )
            if len(questions) == 10:
                return questions
        except Exception:
            pass

        return self._build_fallback_interview_questions(
            candidate_name=candidate_name,
            keywords=keywords,
            evaluation=evaluation_payload,
            normalized_sources=normalized_sources,
        )

    async def submit_interview_answers(
        self,
        candidate_name: str,
        resume_text: str,
        jd_text: str,
        keywords: list[str],
        evaluation: dict | None,
        answers: list[InterviewAnswerInput],
    ) -> dict:
        evaluation_payload = evaluation or {}
        decision = str(evaluation_payload.get("decision") or "待进一步评估")
        match_score = int(evaluation_payload.get("match_score") or 0)
        qa_block = "\n\n".join(
            [
                f"题目ID: {answer.question_id}\n题型: {answer.category}\n问题: {answer.question}\n回答: {answer.answer.strip() or '未作答'}"
                for answer in answers
            ]
        )

        try:
            structured_llm = self.agent.evaluation_llm.with_structured_output(InterviewEvaluationLLMResult)
            prompt = build_interview_submit_prompt(self.agent.system_instruction)
            raw_result = await (prompt | structured_llm).ainvoke(
                {
                    "candidate_name": candidate_name,
                    "keywords": ", ".join(keywords),
                    "jd_text": jd_text,
                    "resume_text": resume_text,
                    "decision": decision,
                    "match_score": match_score,
                    "qa_block": qa_block,
                }
            )
            result: InterviewEvaluationLLMResult = coerce_model(raw_result, InterviewEvaluationLLMResult)
            question_results = [
                {
                    "question_id": item.question_id,
                    "score": item.score,
                    "feedback": item.feedback,
                    "strengths": unique_strings(item.strengths),
                    "improvements": unique_strings(item.improvements),
                }
                for item in result.question_results
            ]
            if len(question_results) != len(answers):
                raise ValueError("question result count mismatch")
            total_score = round(sum(item["score"] for item in question_results) / max(len(question_results), 1))
            return InterviewSubmitResult(
                total_score=total_score,
                verdict=self._build_interview_verdict(total_score),
                overall_feedback=result.overall_feedback,
                strengths=unique_strings(result.strengths),
                risks=unique_strings(result.risks),
                question_results=[InterviewQuestionResult(**item) for item in question_results],
            ).model_dump()
        except Exception:
            return self._build_fallback_interview_result(answers)

    def _build_fallback_interview_questions(
        self,
        candidate_name: str,
        keywords: list[str],
        evaluation: dict,
        normalized_sources: list[dict[str, str]],
    ) -> list[dict]:
        _ = candidate_name
        primary_keyword = keywords[0] if keywords else "当前岗位核心能力"
        secondary_keyword = keywords[1] if len(keywords) > 1 else "项目复杂度"
        risk_texts = [item.get("text", "") for item in evaluation.get("risks", []) if isinstance(item, dict)]
        risk_hint = risk_texts[0] if risk_texts else "请说明你认为自己目前最需要被验证的一项能力。"
        source_ids = [source["source_id"] for source in normalized_sources[:2]]
        return [
            {"question_id": "q1", "category": "technical_depth", "question": f"请你结合最近一段经历，详细讲讲你在 {primary_keyword} 上做过最有挑战的一次技术决策。", "intent": "验证候选人的技术深度与技术判断能力", "source_ids": source_ids},
            {"question_id": "q2", "category": "technical_depth", "question": f"如果让你重新做一次与 {primary_keyword} 相关的核心模块，你会如何优化当时的方案？", "intent": "验证复盘能力与技术取舍能力", "source_ids": source_ids},
            {"question_id": "q3", "category": "technical_depth", "question": f"在你过去的项目里，{secondary_keyword} 相关的问题通常是如何暴露并被解决的？", "intent": "验证问题定位与工程能力", "source_ids": source_ids},
            {"question_id": "q4", "category": "ownership", "question": "请挑一个你最能体现 ownership 的项目，讲清楚你的职责边界、关键决策和最终结果。", "intent": "验证 ownership 与结果导向", "source_ids": source_ids},
            {"question_id": "q5", "category": "ownership", "question": f"如果面试官追问你在项目中的不可替代性，你会如何证明自己而不是团队整体完成了关键部分？", "intent": "验证真实职责与个人贡献", "source_ids": source_ids},
            {"question_id": "q6", "category": "problem_solving", "question": "请举一个你遇到复杂线上问题或业务阻塞的例子，说说你是怎么拆解问题、推动解决的。", "intent": "验证问题解决与推进能力", "source_ids": source_ids},
            {"question_id": "q7", "category": "problem_solving", "question": "如果你入职后发现 JD 中要求的某项能力自己并不完全具备，你会怎么补位并降低风险？", "intent": "验证业务适应与学习策略", "source_ids": []},
            {"question_id": "q8", "category": "communication", "question": "请讲一个你和产品、后端或业务方存在明显分歧，但最终仍推动项目落地的例子。", "intent": "验证沟通协作与影响力", "source_ids": []},
            {"question_id": "q9", "category": "communication", "question": f"你为什么认为自己适合这个岗位？除了技术匹配，还请谈谈你对团队协作和业务节奏的理解。", "intent": "验证岗位动机与表达能力", "source_ids": []},
            {"question_id": "q10", "category": "risk_check", "question": risk_hint, "intent": "验证评估阶段识别出的关键风险点", "source_ids": []},
        ]

    def _build_fallback_interview_result(self, answers: list[InterviewAnswerInput]) -> dict:
        question_results = []
        for answer in answers:
            normalized_answer = answer.answer.strip()
            answer_length = len(normalized_answer)
            if answer_length >= 120:
                score = 78
                feedback = "回答较完整，能够覆盖问题核心，建议进一步补充更量化的结果和细节。"
                strengths = ["基本切题", "表达相对完整"]
                improvements = ["增加数据或结果支撑"]
            elif answer_length >= 50:
                score = 68
                feedback = "回答有一定针对性，但细节和说服力还不够，建议补充实际案例。"
                strengths = ["有基本思路"]
                improvements = ["补充项目细节", "增强结构化表达"]
            elif answer_length > 0:
                score = 55
                feedback = "回答较简略，能看出初步思路，但缺少关键细节与展开。"
                strengths = ["有作答意愿"]
                improvements = ["补充完整案例", "明确问题结论"]
            else:
                score = 20
                feedback = "该题基本未作答，无法判断相关能力。"
                strengths = []
                improvements = ["补充作答内容"]

            question_results.append(
                {
                    "question_id": answer.question_id,
                    "score": score,
                    "feedback": feedback,
                    "strengths": strengths,
                    "improvements": improvements,
                }
            )

        total_score = round(sum(item["score"] for item in question_results) / max(len(question_results), 1))
        return InterviewSubmitResult(
            total_score=total_score,
            verdict=self._build_interview_verdict(total_score),
            overall_feedback="整体回答具备一定岗位匹配基础，但仍建议结合重点题目中的细节、例子与结构化表达进一步判断。",
            strengths=["具备基本作答能力"],
            risks=["部分回答可能缺少足够细节", "建议在线下继续追问关键项目与风险点"],
            question_results=[InterviewQuestionResult(**item) for item in question_results],
        ).model_dump()

    def _build_interview_verdict(self, total_score: int) -> str:
        if total_score >= 75:
            return "通过"
        if total_score >= 60:
            return "待定"
        return "不通过"
