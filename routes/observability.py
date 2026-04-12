from fastapi import APIRouter, Depends
from sqlalchemy import Integer, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.route_helpers import build_observability_filters, build_time_bucket_expr, ensure_observability_enabled
from database import get_db
from models import (
    ModelCallLog,
    ObservabilityLogItemResponse,
    ObservabilityLogsResponse,
    ObservabilitySummaryResponse,
    ObservabilityTrendPoint,
    ObservabilityTrendsResponse,
)


router = APIRouter(prefix="/observability")


@router.get("/summary", summary="观测概览")
async def observability_summary(
    source: str | None = None,
    feature: str | None = None,
    stage: str | None = None,
    model_name: str | None = None,
    prompt_name: str | None = None,
    start_at: str | None = None,
    end_at: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    ensure_observability_enabled()
    filters = build_observability_filters(
        source=source,
        feature=feature,
        stage=stage,
        model_name=model_name,
        prompt_name=prompt_name,
        start_at=start_at,
        end_at=end_at,
    )
    success_case = ModelCallLog.success.cast(Integer)
    fallback_case = ModelCallLog.fallback_used.cast(Integer)
    stmt = select(
        func.count(ModelCallLog.id),
        func.sum(success_case),
        func.sum(fallback_case),
        func.avg(ModelCallLog.latency_ms),
        func.sum(func.coalesce(ModelCallLog.input_tokens, 0)),
        func.sum(func.coalesce(ModelCallLog.output_tokens, 0)),
        func.sum(func.coalesce(ModelCallLog.total_tokens, 0)),
        func.sum(func.coalesce(ModelCallLog.estimated_cost, 0)),
    ).where(*filters)
    row = (await db.execute(stmt)).one()
    total_calls = int(row[0] or 0)
    success_count = int(row[1] or 0)
    fallback_count = int(row[2] or 0)
    return ObservabilitySummaryResponse(
        total_calls=total_calls,
        success_rate=round(success_count / total_calls, 4) if total_calls else 0,
        fallback_rate=round(fallback_count / total_calls, 4) if total_calls else 0,
        avg_latency_ms=round(float(row[3] or 0), 2),
        total_input_tokens=int(row[4] or 0),
        total_output_tokens=int(row[5] or 0),
        total_tokens=int(row[6] or 0),
        total_estimated_cost=round(float(row[7] or 0), 8),
    ).model_dump()


@router.get("/logs", summary="观测日志")
async def observability_logs(
    source: str | None = None,
    feature: str | None = None,
    stage: str | None = None,
    model_name: str | None = None,
    prompt_name: str | None = None,
    start_at: str | None = None,
    end_at: str | None = None,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    ensure_observability_enabled()
    safe_page = max(1, page)
    safe_page_size = max(1, min(100, page_size))
    filters = build_observability_filters(
        source=source,
        feature=feature,
        stage=stage,
        model_name=model_name,
        prompt_name=prompt_name,
        start_at=start_at,
        end_at=end_at,
    )
    total = int((await db.execute(select(func.count(ModelCallLog.id)).where(*filters))).scalar_one() or 0)
    stmt = (
        select(ModelCallLog)
        .where(*filters)
        .order_by(desc(ModelCallLog.created_at))
        .offset((safe_page - 1) * safe_page_size)
        .limit(safe_page_size)
    )
    logs = (await db.execute(stmt)).scalars().all()
    items = [
        ObservabilityLogItemResponse(
            id=log.id,
            request_id=log.request_id,
            source=log.source,
            feature=log.feature,
            stage=log.stage,
            model_name=log.model_name,
            prompt_name=log.prompt_name,
            prompt_version_id=log.prompt_version_id,
            input_summary=log.input_summary or "",
            output_summary=log.output_summary or "",
            input_tokens=log.input_tokens,
            output_tokens=log.output_tokens,
            total_tokens=log.total_tokens,
            latency_ms=log.latency_ms,
            estimated_cost=log.estimated_cost,
            success=log.success,
            fallback_used=log.fallback_used,
            error_message=log.error_message,
            created_at=log.created_at.isoformat(),
        )
        for log in logs
    ]
    return ObservabilityLogsResponse(items=items, total=total, page=safe_page, page_size=safe_page_size).model_dump()


@router.get("/trends", summary="观测趋势")
async def observability_trends(
    source: str | None = None,
    feature: str | None = None,
    stage: str | None = None,
    model_name: str | None = None,
    prompt_name: str | None = None,
    start_at: str | None = None,
    end_at: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    ensure_observability_enabled()
    filters = build_observability_filters(
        source=source,
        feature=feature,
        stage=stage,
        model_name=model_name,
        prompt_name=prompt_name,
        start_at=start_at,
        end_at=end_at,
    )
    bucket_expr = build_time_bucket_expr()
    stmt = (
        select(
            bucket_expr.label("bucket"),
            func.avg(ModelCallLog.latency_ms),
            func.sum(func.coalesce(ModelCallLog.total_tokens, 0)),
            func.sum(func.coalesce(ModelCallLog.estimated_cost, 0)),
            func.count(ModelCallLog.id),
        )
        .where(*filters)
        .group_by(bucket_expr)
        .order_by(bucket_expr)
    )
    points = [
        ObservabilityTrendPoint(
            bucket=str(row[0]),
            latency_ms_avg=round(float(row[1] or 0), 2),
            total_tokens=int(row[2] or 0),
            total_estimated_cost=round(float(row[3] or 0), 8),
            total_calls=int(row[4] or 0),
        )
        for row in (await db.execute(stmt)).all()
    ]
    return ObservabilityTrendsResponse(points=points).model_dump()
