import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.app_context import interview_agent
from core.route_helpers import find_resume
from database import get_db
from models import (
    EvaluationRequest,
    InterviewHistoryItem,
    InterviewHistoryRequest,
    InterviewHistoryResponse,
    InterviewSession,
    InterviewSessionDetailResponse,
    InterviewStartRequest,
    InterviewSubmitRequest,
)


router = APIRouter(prefix="/interview")


@router.post("/start_stream", summary="流式生成模拟面试问题")
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
    resume = await find_resume(evaluation_request, db)
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
                    db=db,
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


@router.post("/submit", summary="提交模拟面试答案并评分")
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
    resume = await find_resume(evaluation_request, db)
    if not resume:
        raise HTTPException(status_code=404, detail="未找到对应的简历")
    if not resume.content:
        raise HTTPException(status_code=409, detail="简历还未解析完成，请稍后再试")

    result = await interview_agent.submit(
        resume=resume,
        jd_text=jd_text,
        jd_keywords=payload.jd_keywords,
        answers=payload.answers,
        db=db,
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


@router.post("/history", summary="查询模拟面试记录")
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


@router.get("/history/{session_id}", summary="查询单次模拟面试详情")
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
