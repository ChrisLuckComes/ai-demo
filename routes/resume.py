import asyncio
import hashlib
import json
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.app_context import agent, interview_agent, logger
from core.route_helpers import find_resume
from database import get_db, redis_client
from models import ChatRequest, EvaluationRequest, JDAnalysisRequest, QueryRequest, Resume, ResumeStatus


router = APIRouter()


@router.post("/analyze_jd", summary="分析JD并提取关键词")
async def analyze_jd(request: JDAnalysisRequest):
    request_started_at = time.perf_counter()
    jd_text = request.jd_text.strip()
    if not jd_text:
        logger.info("analyze_jd skipped_empty elapsed_ms=%d", round((time.perf_counter() - request_started_at) * 1000))
        return {"keywords": []}

    normalized_jd_text = " ".join(jd_text.split())
    text_length = len(normalized_jd_text)
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
        await redis_client.setex(cache_key, 3600, json.dumps(result, ensure_ascii=False))
    logger.info(
        "analyze_jd completed elapsed_ms=%d extraction_ms=%d text_length=%d keyword_count=%d",
        round((time.perf_counter() - request_started_at) * 1000),
        extraction_elapsed_ms,
        text_length,
        len(result.get("keywords", [])),
    )
    return result


@router.post("/ocr_jd_image", summary="图片OCR提取JD文字")
async def ocr_jd_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持图片文件进行OCR")

    extracted = await agent.extract_text_from_image(
        file_bytes=await file.read(),
        mime_type=file.content_type,
    )
    return extracted.model_dump()


@router.post("/upload_resume", summary="上传并解析简历")
async def upload_resume(
    file: UploadFile = File(...),
    candidate_name: str = Form(...),
    phone: str = Form(...),
    user_id: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith((".docx", ".pdf")):
        raise HTTPException(status_code=400, detail="仅支持 .docx 或 .pdf 格式的文件")

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
        resume.status = ResumeStatus.FAILED
        await db.flush()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": "简历上传并解析完成",
        "resume_id": resume.id,
        "status": resume.status.value,
    }


@router.get("/resumes", summary="查询已上传简历列表")
async def list_resumes(user_id: str, db: AsyncSession = Depends(get_db)):
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


@router.delete("/resumes/{resume_id}", summary="删除已上传简历")
async def delete_resume(resume_id: int, user_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(Resume).where(Resume.id == resume_id, Resume.user_id == user_id)
    resume = (await db.execute(stmt)).scalar_one_or_none()
    if not resume:
        raise HTTPException(status_code=404, detail="未找到对应的简历")

    agent.delete_resume_vectors(user_id=user_id, resume_id=resume.id, candidate_name=resume.candidate_name)
    await db.delete(resume)
    await db.flush()
    return {"message": "简历已删除", "resume_id": resume_id}


@router.post("/evaluate", summary="评估简历")
async def evaluate_resume(request: EvaluationRequest, db: AsyncSession = Depends(get_db)):
    jd_text = request.jd_text.strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="`jd_text` 不能为空")
    if not request.jd_keywords:
        raise HTTPException(status_code=400, detail="请先分析JD")

    resume = await find_resume(request, db)
    if not resume:
        raise HTTPException(status_code=404, detail="未找到对应的简历")
    if not resume.content:
        raise HTTPException(status_code=409, detail="简历还未解析完成，请稍后再试")

    resume.status = ResumeStatus.EVALUATING
    await db.flush()

    evaluation = await agent.evaluate_resume(
        resume_text=resume.content,
        jd_text=jd_text,
        db=db,
        jd_keywords=request.jd_keywords,
    )
    resume.evaluation_result = evaluation
    resume.status = ResumeStatus.COMPLETED
    await db.flush()
    return {"evaluation": evaluation}


@router.post("/evaluate_stream", summary="流式评估简历")
async def evaluate_resume_stream(request: EvaluationRequest, db: AsyncSession = Depends(get_db)):
    jd_text = request.jd_text.strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="`jd_text` 不能为空")
    if not request.jd_keywords:
        raise HTTPException(status_code=400, detail="请先分析JD")

    resume = await find_resume(request, db)
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
                db=db,
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
                db=db,
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
                db=db,
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
                db=db,
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
                        db=db,
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


@router.post("/query", summary="非流式查询")
async def query_resume(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    reply = await agent.ask(
        question=request.text,
        user_id=request.user_id,
        db=db,
        candidate_name=request.candidate_name,
        resume_id=request.resume_id,
    )
    sources = await agent.get_chat_sources(
        question=request.text,
        user_id=request.user_id,
        candidate_name=request.candidate_name,
        resume_id=request.resume_id,
    )
    return {"reply": reply, "sources": sources}


@router.post("/chat", summary="流式聊天")
async def chat_endpoint(payload: ChatRequest, db: AsyncSession = Depends(get_db)):
    async def event_stream():
        full_response_parts: list[str] = []
        try:
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

            suggestions = await agent.generate_follow_up_suggestions(
                question=payload.text,
                answer="".join(full_response_parts),
                db=db,
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
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
