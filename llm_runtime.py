import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.runnables import Runnable
from sqlalchemy.ext.asyncio import AsyncSession

from llm_costs import estimate_cost
from models import ModelCallLog


def build_preview(value: Any, limit: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
    normalized = " ".join(text.split())
    return normalized[:limit]


def summarize_payload(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return f"len={len(value)} preview={build_preview(value)}"
    if isinstance(value, dict):
        keys = ", ".join(sorted(value.keys()))
        return f"dict_keys=[{keys}] preview={build_preview(value)}"
    if isinstance(value, list):
        return f"list_len={len(value)} preview={build_preview(value)}"
    return build_preview(value)


def _sanitize_usage_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in candidate.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = [
                item for item in value if isinstance(item, (str, int, float, bool)) or item is None
            ]
        elif isinstance(value, dict):
            sanitized[key] = {
                nested_key: nested_value
                for nested_key, nested_value in value.items()
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None
            }
        else:
            sanitized[key] = str(value)
    return sanitized


def extract_usage_details(raw_response: Any) -> dict[str, Any]:
    details: dict[str, Any] = {
        "provider": "google_genai",
        "raw_candidates": [],
    }
    if raw_response is None:
        return details

    usage_metadata = getattr(raw_response, "usage_metadata", None)
    response_metadata = getattr(raw_response, "response_metadata", None)
    if isinstance(usage_metadata, dict):
        details["raw_candidates"].append({
            "source": "usage_metadata",
            "value": _sanitize_usage_candidate(usage_metadata),
        })
    if isinstance(response_metadata, dict):
        for key in ("usage_metadata", "token_usage", "usage"):
            candidate = response_metadata.get(key)
            if isinstance(candidate, dict):
                details["raw_candidates"].append({
                    "source": f"response_metadata.{key}",
                    "value": _sanitize_usage_candidate(candidate),
                })
        details["response_metadata_preview"] = _sanitize_usage_candidate(response_metadata)
    return details


def extract_usage_metrics(raw_response: Any) -> dict[str, Optional[int]]:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    if raw_response is None:
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    candidates: list[Any] = []
    usage_metadata = getattr(raw_response, "usage_metadata", None)
    response_metadata = getattr(raw_response, "response_metadata", None)
    if usage_metadata:
        candidates.append(usage_metadata)
    if response_metadata and isinstance(response_metadata, dict):
        candidates.extend(
            [
                response_metadata.get("usage_metadata"),
                response_metadata.get("token_usage"),
                response_metadata.get("usage"),
            ]
        )

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        input_tokens = candidate.get("input_tokens") or candidate.get("prompt_token_count") or candidate.get("prompt_tokens")
        output_tokens = candidate.get("output_tokens") or candidate.get("candidates_token_count") or candidate.get("completion_tokens")
        total_tokens = candidate.get("total_tokens") or candidate.get("total_token_count")
        if input_tokens is not None:
            input_tokens = int(input_tokens)
        if output_tokens is not None:
            output_tokens = int(output_tokens)
        if total_tokens is not None:
            total_tokens = int(total_tokens)
        if input_tokens is not None or output_tokens is not None or total_tokens is not None:
            break

    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


@dataclass
class LLMRunResult:
    request_id: str
    log_id: Optional[int]
    parsed_output: Any
    raw_output: Any
    latency_ms: int
    usage: dict[str, Optional[int]]
    estimated_cost: Optional[float]
    success: bool
    fallback_used: bool
    error_message: Optional[str]
    usage_details: dict[str, Any]


class LLMRuntime:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def invoke_structured(
        self,
        *,
        runnable: Runnable,
        payload: dict[str, Any],
        model_name: str,
        source: str,
        feature: str,
        stage: str,
        prompt_name: str,
        prompt_version_id: Optional[int] = None,
        request_id: Optional[str] = None,
        fallback_used: bool = False,
        extra_json: Optional[dict[str, Any]] = None,
    ) -> LLMRunResult:
        normalized_request_id = request_id or str(uuid.uuid4())
        started_at = time.perf_counter()
        try:
            raw_result = await runnable.ainvoke(payload)
            latency_ms = round((time.perf_counter() - started_at) * 1000)
            parsed_output = raw_result.get("parsed") if isinstance(raw_result, dict) else raw_result
            raw_output = raw_result.get("raw") if isinstance(raw_result, dict) else raw_result
            usage = extract_usage_metrics(raw_output)
            usage_details = extract_usage_details(raw_output)
            estimated_cost = estimate_cost(
                model_name,
                usage.get("input_tokens"),
                usage.get("output_tokens"),
            )
            merged_extra_json = dict(extra_json or {})
            merged_extra_json["usage_details"] = usage_details
            log = ModelCallLog(
                request_id=normalized_request_id,
                source=source,
                feature=feature,
                stage=stage,
                model_name=model_name,
                prompt_name=prompt_name,
                prompt_version_id=prompt_version_id,
                input_summary=summarize_payload(payload),
                output_summary=summarize_payload(parsed_output),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
                latency_ms=latency_ms,
                estimated_cost=estimated_cost,
                success=True,
                fallback_used=fallback_used,
                error_message=None,
                extra_json=merged_extra_json,
            )
            self.db.add(log)
            await self.db.flush()
            return LLMRunResult(
                request_id=normalized_request_id,
                log_id=log.id,
                parsed_output=parsed_output,
                raw_output=raw_output,
                latency_ms=latency_ms,
                usage=usage,
                estimated_cost=estimated_cost,
                success=True,
                fallback_used=fallback_used,
                error_message=None,
                usage_details=usage_details,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started_at) * 1000)
            merged_extra_json = dict(extra_json or {})
            log = ModelCallLog(
                request_id=normalized_request_id,
                source=source,
                feature=feature,
                stage=stage,
                model_name=model_name,
                prompt_name=prompt_name,
                prompt_version_id=prompt_version_id,
                input_summary=summarize_payload(payload),
                output_summary="",
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                latency_ms=latency_ms,
                estimated_cost=None,
                success=False,
                fallback_used=fallback_used,
                error_message=str(exc),
                extra_json=merged_extra_json,
            )
            self.db.add(log)
            await self.db.flush()
            raise
