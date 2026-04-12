from datetime import datetime, timedelta

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession

from core.app_context import OBSERVABILITY_ENABLED, PLAYGROUND_ENABLED
from database import engine
from models import EvaluationRequest, ModelCallLog, Resume


def ensure_observability_enabled() -> None:
    if not OBSERVABILITY_ENABLED:
        raise HTTPException(status_code=404, detail="observability disabled")


def ensure_playground_enabled() -> None:
    if not PLAYGROUND_ENABLED:
        raise HTTPException(status_code=404, detail="prompt playground disabled")


def normalize_datetime(value: str | None, default: datetime) -> datetime:
    if not value:
        return default
    return datetime.fromisoformat(value)


def build_observability_filters(
    *,
    source: str | None,
    feature: str | None,
    stage: str | None,
    model_name: str | None,
    prompt_name: str | None,
    start_at: str | None,
    end_at: str | None,
):
    now = datetime.now()
    start_dt = normalize_datetime(start_at, now - timedelta(days=7))
    end_dt = normalize_datetime(end_at, now)
    clauses = [ModelCallLog.created_at >= start_dt, ModelCallLog.created_at <= end_dt]
    if source:
        clauses.append(ModelCallLog.source == source)
    if feature:
        clauses.append(ModelCallLog.feature == feature)
    if stage:
        clauses.append(ModelCallLog.stage == stage)
    if model_name:
        clauses.append(ModelCallLog.model_name == model_name)
    if prompt_name:
        clauses.append(ModelCallLog.prompt_name == prompt_name)
    return clauses


def build_time_bucket_expr():
    drivername = make_url(str(engine.url)).drivername
    if drivername.startswith("postgresql"):
        return func.to_char(ModelCallLog.created_at, "YYYY-MM-DD")
    return func.strftime("%Y-%m-%d", ModelCallLog.created_at)


async def find_resume(request: EvaluationRequest, db: AsyncSession) -> Resume | None:
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
