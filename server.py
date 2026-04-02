import asyncio
import hashlib
import inspect
import json
import logging
import time

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent import ResumeAgent
from database import Base, engine, get_db, redis_client
from interview_agent import InterviewAgent
from models import (
    ChatRequest,
    EvaluationRequest,
    InterviewStartRequest,
    InterviewSubmitRequest,
    InterviewHistoryRequest,
    InterviewHistoryResponse,
    InterviewHistoryItem,
    InterviewSessionDetailResponse,
    InterviewSession,
    JDAnalysisRequest,
    QueryRequest,
    Resume,
    ResumeStatus,
)


async def _close_redis() -> None:
    if not redis_client:
        return
    # 兼容不同 redis 客户端版本的关闭方法，避免应用退出时资源泄漏。
    close_method = getattr(redis_client, "aclose", None) or getattr(redis_client, "close", None)
    if close_method is None:
        return
    result = close_method()
    if inspect.isawaitable(result):
        await result


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 应用启动时建表，关闭时释放数据库和 Redis 连接。
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()
    await _close_redis()


app = FastAPI(lifespan=lifespan)
agent = ResumeAgent(redis_client)
interview_agent = InterviewAgent(agent=agent, redis_client=redis_client)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "resume-agent backend is running",
        "endpoints": [
            "/analyze_jd",
            "/ocr_jd_image",
            "/upload_resume",
            "/resumes",
            "/evaluate",
            "/chat",
        ],
    }


@app.post("/analyze_jd", summary="分析JD并提取关键词")
async def analyze_jd(request: JDAnalysisRequest):
    request_started_at = time.perf_counter()
    jd_text = request.jd_text.strip()
    if not jd_text:
        logger.info("analyze_jd skipped_empty elapsed_ms=%d", round((time.perf_counter() - request_started_at) * 1000))
        return {"keywords": []}

    normalized_jd_text = " ".join(jd_text.split())
    text_length = len(normalized_jd_text)

    # 相同 JD 的关键词提取结果会缓存，避免重复调用模型。
    cache_key = f"jd_analysis:{hashlib.md5(normalized_jd_text.encode('utf-8')).hexdigest()}"
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            payload = json.loads(cached)
            logger.info(
                "analyze_jd cache_hit elapsed_ms=%d text_length=%d keyword_count=%d",
                round((time.perf_counter() - request_started_at) * 1000),
                text_length,
                len(payload.get("keywords", [])),
            )
            return payload

    logger.info("analyze_jd cache_miss text_length=%d", text_length)
    extraction_started_at = time.perf_counter()
    result = (await agent.analyze_jd(normalized_jd_text)).model_dump()
    extraction_elapsed_ms = round((time.perf_counter() - extraction_started_at) * 1000)
    if redis_client:
        await redis_client.setex(
            cache_key,
            3600,
            json.dumps(result, ensure_ascii=False),
        )
    logger.info(
        "analyze_jd completed elapsed_ms=%d extraction_ms=%d text_length=%d keyword_count=%d",
        round((time.perf_counter() - request_started_at) * 1000),
        extraction_elapsed_ms,
        text_length,
        len(result.get("keywords", [])),
    )
    return result


@app.post("/ocr_jd_image", summary="图片OCR提取JD文字")
async def ocr_jd_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持图片文件进行OCR")

    # 这里只负责校验与转发，真正的 OCR 细节都收敛在 agent 内部。
    extracted = await agent.extract_text_from_image(
        file_bytes=await file.read(),
        mime_type=file.content_type,
    )
    return extracted.model_dump()


@app.post("/upload_resume", summary="上传并解析简历")
async def upload_resume(
    file: UploadFile = File(...),
    candidate_name: str = Form(...),
    phone: str = Form(...),
    user_id: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith((".docx", ".pdf")):
        raise HTTPException(status_code=400, detail="仅支持 .docx 或 .pdf 格式的文件")

    # 当前用 user_id + phone 识别同一候选人，重复上传时走覆盖更新而不是新增。
    stmt = select(Resume).where(Resume.user_id == user_id, Resume.phone == phone)
    existing_resume = (await db.execute(stmt)).scalar_one_or_none()

    if existing_resume:
        resume = existing_resume
        resume.candidate_name = candidate_name
        resume.status = ResumeStatus.PENDING
        resume.content = None
        resume.evaluation_result = None
    else:
        resume = Resume(
            user_id=user_id,
            candidate_name=candidate_name,
            phone=phone,
            status=ResumeStatus.PENDING,
        )
        db.add(resume)

    await db.flush()

    try:
        # ingest_resume 会完成解析、切块、向量化入库，是上传链路的核心步骤。
        resume.status = ResumeStatus.PARSING
        raw_text = await agent.ingest_resume(
            file_name=file.filename,
            file_content=await file.read(),
            user_id=user_id,
            resume_id=resume.id,
            candidate_name=candidate_name,
            phone=phone,
        )
        resume.content = raw_text
        resume.status = ResumeStatus.COMPLETED
        await db.flush()
    except Exception as exc:
        # 一旦解析或入库失败，状态回写 FAILED，便于前端展示和后续排查。
        resume.status = ResumeStatus.FAILED
        await db.flush()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": "简历上传并解析完成",
        "resume_id": resume.id,
        "status": resume.status.value,
    }


@app.get("/resumes", summary="查询已上传简历列表")
async def list_resumes(user_id: str, db: AsyncSession = Depends(get_db)):
    # 简历列表按最近更新时间倒序，方便前端优先展示最新上传或最新评估的数据。
    stmt = select(Resume).where(Resume.user_id == user_id).order_by(desc(Resume.updated_at))
    resumes = (await db.execute(stmt)).scalars().all()
    return {
        "items": [
            {
                "resume_id": resume.id,
                "candidate_name": resume.candidate_name,
                "phone": resume.phone,
                "status": resume.status.value,
                "updated_at": resume.updated_at.isoformat(),
            }
            for resume in resumes
        ]
    }


@app.delete("/resumes/{resume_id}", summary="删除已上传简历")
async def delete_resume(resume_id: int, user_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Resume).where(Resume.id == resume_id, Resume.user_id == user_id)
    resume = (await db.execute(stmt)).scalar_one_or_none()
    if not resume:
        raise HTTPException(status_code=404, detail="未找到对应的简历")

    # 删除数据库记录前先删除向量库中的分片，避免残留脏检索数据。
    agent.delete_resume_vectors(
        user_id=user_id,
        resume_id=resume.id,
        candidate_name=resume.candidate_name,
    )
    await db.delete(resume)
    await db.flush()
    return {"message": "简历已删除", "resume_id": resume_id}


@app.post("/evaluate", summary="评估简历")
async def evaluate_resume(request: EvaluationRequest, db: AsyncSession = Depends(get_db)):
    jd_text = request.jd_text.strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="`jd_text` 不能为空")
    if not request.jd_keywords:
        raise HTTPException(status_code=400, detail="请先分析JD")

    # 同一套评估接口支持按 resume_id / phone / candidate_name 三种方式定位简历。
    resume = await _find_resume(request, db)
    if not resume:
        raise HTTPException(status_code=404, detail="未找到对应的简历")
    if not resume.content:
        raise HTTPException(status_code=409, detail="简历还未解析完成，请稍后再试")

    # 评估结果除了分数和文本结论，还会带来源片段，供前端展示可解释性。
    resume.status = ResumeStatus.EVALUATING
    await db.flush()

    evaluation = await agent.evaluate_resume(
        resume_text=resume.content,
        jd_text=jd_text,
        jd_keywords=request.jd_keywords,
    )
    resume.evaluation_result = evaluation
    resume.status = ResumeStatus.COMPLETED
    await db.flush()
    return {"evaluation": evaluation}


@app.post("/evaluate_stream", summary="流式评估简历")
async def evaluate_resume_stream(request: EvaluationRequest, db: AsyncSession = Depends(get_db)):
    jd_text = request.jd_text.strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="`jd_text` 不能为空")
    if not request.jd_keywords:
        raise HTTPException(status_code=400, detail="请先分析JD")

    resume = await _find_resume(request, db)
    if not resume:
        raise HTTPException(status_code=404, detail="未找到对应的简历")
    if not resume.content:
        raise HTTPException(status_code=409, detail="简历还未解析完成，请稍后再试")

    async def event_stream():
        stream_started_at = time.perf_counter()
        resume.status = ResumeStatus.EVALUATING
        await db.flush()
        resume_text = resume.content or ""

        try:
            yield f"data: {json.dumps({'type': 'phase', 'phase': 'preparing'}, ensure_ascii=False)}\n\n"
            sources = agent.build_evaluation_sources(resume_text)
            normalized_sources = [dict(source) for source in sources]
            radar_metrics = []

            logger.info(
                "evaluate_stream sources_ready resume_id=%s elapsed_ms=%d source_count=%d",
                resume.id,
                round((time.perf_counter() - stream_started_at) * 1000),
                len(normalized_sources),
            )

            yield f"data: {json.dumps({'type': 'phase', 'phase': 'sources'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': normalized_sources}, ensure_ascii=False)}\n\n"

            keywords = agent.require_jd_keywords(request.jd_keywords)

            yield f"data: {json.dumps({'type': 'phase', 'phase': 'scoring'}, ensure_ascii=False)}\n\n"
            scoring_started_at = time.perf_counter()
            score_result = await agent.generate_evaluation_score(
                keywords=keywords,
                resume_text=resume_text,
                jd_text=jd_text,
            )
            logger.info(
                "evaluate_stream scoring_ready resume_id=%s elapsed_ms=%d stage_ms=%d",
                resume.id,
                round((time.perf_counter() - stream_started_at) * 1000),
                round((time.perf_counter() - scoring_started_at) * 1000),
            )
            yield f"data: {json.dumps({'type': 'score', 'match_score': score_result['match_score'], 'title': score_result['title']}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'phase', 'phase': 'radar'}, ensure_ascii=False)}\n\n"
            radar_metrics = agent.build_radar_payload(score_result['match_score'])
            yield f"data: {json.dumps({'type': 'radar_metrics', 'radar_metrics': radar_metrics}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'phase', 'phase': 'summary'}, ensure_ascii=False)}\n\n"
            summary_started_at = time.perf_counter()
            summary_result = await agent.generate_evaluation_summary(
                resume_text=resume_text,
                jd_text=jd_text,
                keywords=keywords,
                sources=sources,
                match_score=score_result["match_score"],
                decision=score_result["decision"],
            )
            logger.info(
                "evaluate_stream summary_ready resume_id=%s elapsed_ms=%d stage_ms=%d",
                resume.id,
                round((time.perf_counter() - stream_started_at) * 1000),
                round((time.perf_counter() - summary_started_at) * 1000),
            )

            yield f"data: {json.dumps({'type': 'summary', 'summary': summary_result['summary'], 'summary_source_ids': summary_result['summary_source_ids']}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'phase', 'phase': 'highlights'}, ensure_ascii=False)}\n\n"
            highlights_started_at = time.perf_counter()
            highlights_result = await agent.generate_evaluation_items(
                item_type='highlights',
                resume_text=resume_text,
                jd_text=jd_text,
                keywords=keywords,
                sources=sources,
                match_score=score_result['match_score'],
                decision=score_result['decision'],
                summary=summary_result['summary'],
            )
            logger.info(
                "evaluate_stream highlights_ready resume_id=%s elapsed_ms=%d stage_ms=%d",
                resume.id,
                round((time.perf_counter() - stream_started_at) * 1000),
                round((time.perf_counter() - highlights_started_at) * 1000),
            )
            yield f"data: {json.dumps({'type': 'highlights', 'highlights': highlights_result['items']}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'phase', 'phase': 'risks'}, ensure_ascii=False)}\n\n"
            risks_started_at = time.perf_counter()
            risks_result = await agent.generate_evaluation_items(
                item_type='risks',
                resume_text=resume_text,
                jd_text=jd_text,
                keywords=keywords,
                sources=sources,
                match_score=score_result['match_score'],
                decision=score_result['decision'],
                summary=summary_result['summary'],
            )
            logger.info(
                "evaluate_stream risks_ready resume_id=%s elapsed_ms=%d stage_ms=%d",
                resume.id,
                round((time.perf_counter() - stream_started_at) * 1000),
                round((time.perf_counter() - risks_started_at) * 1000),
            )
            yield f"data: {json.dumps({'type': 'risks', 'risks': risks_result['items']}, ensure_ascii=False)}\n\n"
            
            yield f"data: {json.dumps({'type': 'phase', 'phase': 'finalizing'}, ensure_ascii=False)}\n\n"

            # 最终结果直接复用已逐步生成的各模块，避免再次调用模型导致同一份 JD/简历出现二次漂移。
            evaluation = {
                'title': score_result['title'],
                'decision': score_result['decision'],
                'match_score': score_result['match_score'],
                'radar_metrics': radar_metrics,
                'summary': summary_result['summary'],
                'summary_source_ids': summary_result['summary_source_ids'],
                'highlights': highlights_result['items'],
                'risks': risks_result['items'],
                'sources': normalized_sources,
            }

            resume.evaluation_result = evaluation
            resume.status = ResumeStatus.COMPLETED
            await db.flush()

            if redis_client:
                asyncio.create_task(
                    interview_agent.warm(
                        user_id=request.user_id,
                        resume=resume,
                        jd_text=jd_text,
                        jd_keywords=request.jd_keywords,
                    )
                )

            logger.info(
                "evaluate_stream completed resume_id=%s elapsed_ms=%d",
                resume.id,
                round((time.perf_counter() - stream_started_at) * 1000),
            )

            yield f"data: {json.dumps({'type': 'result', 'evaluation': evaluation}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            resume.status = ResumeStatus.FAILED
            await db.flush()
            logger.exception(
                "evaluate_stream failed resume_id=%s elapsed_ms=%d",
                resume.id,
                round((time.perf_counter() - stream_started_at) * 1000),
            )
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query", summary="非流式查询")
async def query_resume(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    # 非流式接口适合脚本或简单调用，内部仍然复用 agent 的聊天能力。
    reply = await agent.ask(
        question=request.text,
        user_id=request.user_id,
        db=db,
        candidate_name=request.candidate_name,
        resume_id=request.resume_id,
    )
    # 额外返回检索证据，便于调用方展示“回答依据来自哪里”。
    sources = await agent.get_chat_sources(
        question=request.text,
        user_id=request.user_id,
        candidate_name=request.candidate_name,
        resume_id=request.resume_id,
    )
    return {"reply": reply, "sources": sources}


@app.post("/chat", summary="流式聊天")
async def chat_endpoint(payload: ChatRequest, db: AsyncSession = Depends(get_db)):
    async def event_stream():
        full_response_parts: list[str] = []
        try:
            # 先逐块把模型输出推给前端，形成真实的流式聊天体验。
            async for chunk in agent.stream_chat(
                question=payload.text,
                user_id=payload.user_id,
                db=db,
                candidate_name=payload.candidate_name,
                resume_id=payload.resume_id,
            ):
                if not chunk:
                    continue
                full_response_parts.append(chunk)
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"

            # 文本流结束后，再补发建议追问和来源片段，前端可以分开渲染。
            suggestions = await agent.generate_follow_up_suggestions(
                question=payload.text,
                answer="".join(full_response_parts),
                candidate_name=payload.candidate_name,
            )
            sources = await agent.get_chat_sources(
                question=payload.text,
                user_id=payload.user_id,
                candidate_name=payload.candidate_name,
                resume_id=payload.resume_id,
            )
            yield f"data: {json.dumps({'type': 'suggestions', 'items': suggestions}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'items': sources}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            # SSE 场景不能直接抛异常，需要转成 error 事件通知前端。
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/interview/start_stream", summary="流式生成模拟面试问题")
async def interview_start_stream(payload: InterviewStartRequest, db: AsyncSession = Depends(get_db)):
    jd_text = payload.jd_text.strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="`jd_text` 不能为空")
    if not payload.jd_keywords:
        raise HTTPException(status_code=400, detail="请先分析JD")
    interview_identity = payload.interview_identity.strip()
    if not interview_identity:
        raise HTTPException(status_code=400, detail="`interview_identity` 不能为空")

    evaluation_request = EvaluationRequest(
        user_id=payload.user_id,
        jd_text=payload.jd_text,
        resume_id=payload.resume_id,
        candidate_name=payload.candidate_name,
        phone=payload.phone,
        jd_keywords=payload.jd_keywords,
    )
    resume = await _find_resume(evaluation_request, db)
    if not resume:
        raise HTTPException(status_code=404, detail="未找到对应的简历")
    if not resume.content:
        raise HTTPException(status_code=409, detail="简历还未解析完成，请稍后再试")
    async def event_stream():
        try:
            yield f"data: {json.dumps({'type': 'phase', 'phase': 'preparing'}, ensure_ascii=False)}\n\n"
            cache_key = interview_agent.build_cache_key(payload.user_id, resume, jd_text)
            cached_questions = await interview_agent.get_cached_questions(cache_key)
            if cached_questions:
                questions = cached_questions
            else:
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'generating'}, ensure_ascii=False)}\n\n"
                questions = await interview_agent.prepare(
                    user_id=payload.user_id,
                    resume=resume,
                    jd_text=jd_text,
                    jd_keywords=payload.jd_keywords,
                )
            stmt = select(InterviewSession).where(
                InterviewSession.user_id == payload.user_id,
                InterviewSession.resume_id == resume.id,
                InterviewSession.interview_identity == interview_identity,
                InterviewSession.status == "draft",
            )
            session = (await db.execute(stmt)).scalar_one_or_none()
            if session is None:
                session = InterviewSession(
                    user_id=payload.user_id,
                    resume_id=resume.id,
                    candidate_name=resume.candidate_name,
                    interview_identity=interview_identity,
                    status="draft",
                    questions=questions,
                    answers=[],
                    result=None,
                )
                db.add(session)
            else:
                session.candidate_name = resume.candidate_name
                session.questions = questions
                session.answers = []
                session.result = None
            await db.flush()
            yield f"data: {json.dumps({'type': 'session', 'session_id': session.session_id}, ensure_ascii=False)}\n\n"
            for question in questions:
                yield f"data: {json.dumps({'type': 'question', 'question': question}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/interview/submit", summary="提交模拟面试答案并评分")
async def interview_submit(payload: InterviewSubmitRequest, db: AsyncSession = Depends(get_db)):
    jd_text = payload.jd_text.strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="`jd_text` 不能为空")
    if not payload.jd_keywords:
        raise HTTPException(status_code=400, detail="请先分析JD")
    interview_identity = payload.interview_identity.strip()
    if not interview_identity:
        raise HTTPException(status_code=400, detail="`interview_identity` 不能为空")
    if len(payload.answers) != 10:
        raise HTTPException(status_code=400, detail="模拟面试需要提交 10 道题的答案")

    evaluation_request = EvaluationRequest(
        user_id=payload.user_id,
        jd_text=payload.jd_text,
        resume_id=payload.resume_id,
        candidate_name=payload.candidate_name,
        phone=payload.phone,
        jd_keywords=payload.jd_keywords,
    )
    resume = await _find_resume(evaluation_request, db)
    if not resume:
        raise HTTPException(status_code=404, detail="未找到对应的简历")
    if not resume.content:
        raise HTTPException(status_code=409, detail="简历还未解析完成，请稍后再试")
    result = await interview_agent.submit(
        resume=resume,
        jd_text=jd_text,
        jd_keywords=payload.jd_keywords,
        answers=payload.answers,
    )
    session = None
    if payload.session_id:
        stmt = select(InterviewSession).where(
            InterviewSession.session_id == payload.session_id,
            InterviewSession.user_id == payload.user_id,
        )
        session = (await db.execute(stmt)).scalar_one_or_none()
    if session is None:
        session = InterviewSession(
            user_id=payload.user_id,
            resume_id=resume.id,
            candidate_name=resume.candidate_name,
            interview_identity=interview_identity,
            status="completed",
            questions=[answer.model_dump(exclude={"answer"}) for answer in payload.answers],
            answers=[answer.model_dump() for answer in payload.answers],
            result=result,
        )
        db.add(session)
    else:
        session.interview_identity = interview_identity
        session.status = "completed"
        session.questions = [answer.model_dump(exclude={"answer"}) for answer in payload.answers]
        session.answers = [answer.model_dump() for answer in payload.answers]
        session.result = result
    await db.flush()
    return result


@app.post("/interview/history", summary="查询模拟面试记录")
async def interview_history(payload: InterviewHistoryRequest, db: AsyncSession = Depends(get_db)):
    interview_identity = payload.interview_identity.strip()
    if not interview_identity:
        raise HTTPException(status_code=400, detail="`interview_identity` 不能为空")

    stmt = select(InterviewSession).where(
        InterviewSession.user_id == payload.user_id,
        InterviewSession.interview_identity == interview_identity,
    ).order_by(desc(InterviewSession.created_at))
    if payload.resume_id is not None:
        stmt = stmt.where(InterviewSession.resume_id == payload.resume_id)

    sessions = (await db.execute(stmt)).scalars().all()
    items = []
    for session in sessions:
        result = session.result or {}
        items.append(
            InterviewHistoryItem(
                session_id=session.session_id,
                interview_identity=session.interview_identity,
                candidate_name=session.candidate_name or "候选人",
                verdict=str(result.get("verdict") or "待定"),
                total_score=max(0, min(100, int(result.get("total_score") or 0))),
                created_at=session.created_at.isoformat(),
            )
        )
    return InterviewHistoryResponse(items=items).model_dump()


@app.get("/interview/history/{session_id}", summary="查询单次模拟面试详情")
async def interview_history_detail(session_id: str, user_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(InterviewSession).where(
        InterviewSession.session_id == session_id,
        InterviewSession.user_id == user_id,
    )
    session = (await db.execute(stmt)).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="未找到对应的模拟面试记录")

    return InterviewSessionDetailResponse(
        session_id=session.session_id,
        interview_identity=session.interview_identity,
        candidate_name=session.candidate_name or "候选人",
        status=session.status,
        questions=session.questions,
        answers=session.answers,
        result=session.result or {},
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    ).model_dump()


async def _find_resume(request: EvaluationRequest, db: AsyncSession) -> Resume | None:
    # 优先使用最精确的 resume_id，其次回退到 phone / candidate_name。
    if request.resume_id is not None:
        stmt = select(Resume).where(
            Resume.id == request.resume_id,
            Resume.user_id == request.user_id,
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    if request.phone:
        stmt = select(Resume).where(
            Resume.user_id == request.user_id,
            Resume.phone == request.phone,
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    if request.candidate_name:
        stmt = select(Resume).where(
            Resume.user_id == request.user_id,
            Resume.candidate_name == request.candidate_name,
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    return None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
