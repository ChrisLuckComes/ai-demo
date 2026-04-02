import re
from typing import Any, Iterable, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from models import EvidenceSource, EvaluationItem, RadarMetric


ModelT = TypeVar("ModelT", bound=BaseModel)


def coerce_model(value: Any, model_type: type[ModelT]) -> ModelT:
    if isinstance(value, model_type):
        return value
    if isinstance(value, dict):
        return model_type.model_validate(value)
    if hasattr(value, "model_dump"):
        return model_type.model_validate(value.model_dump())
    return model_type.model_validate(value)


def to_langchain_message(role: str, content: str) -> BaseMessage:
    if role == "user":
        return HumanMessage(content=content)
    return AIMessage(content=content)


def unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def fallback_keywords(text: str) -> list[str]:
    stopwords = {
        "负责",
        "熟悉",
        "能力",
        "经验",
        "优先",
        "相关",
        "以上",
        "具备",
        "参与",
        "本科",
        "大专",
        "岗位",
        "职位",
        "工作",
        "公司",
        "团队",
        "产品",
        "业务",
        "要求",
        "能够",
        "我们",
        "你将",
    }
    library = [
        "Python",
        "Java",
        "Go",
        "C++",
        "JavaScript",
        "TypeScript",
        "React",
        "Vue",
        "Node.js",
        "SQL",
        "MySQL",
        "PostgreSQL",
        "Redis",
        "Kafka",
        "Docker",
        "Kubernetes",
        "LangChain",
        "RAG",
        "LLM",
        "AI",
        "机器学习",
        "深度学习",
        "数据分析",
        "项目管理",
        "沟通协作",
    ]
    matched_library = [item for item in library if item.lower() in text.lower()]
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{1,}|[\u4e00-\u9fff]{2,8}", text)
    keywords = [token for token in tokens if token not in stopwords and len(token) >= 2]
    return unique_strings([*matched_library, *keywords])[:8]


def normalize_keyword_candidates(candidates: Iterable[str], text: str) -> list[str]:
    normalized: list[str] = []
    for candidate in candidates:
        keyword = candidate.strip().strip("，,。；;:：()[]{}")
        if len(keyword) < 2 or len(keyword) > 30:
            continue

        match = re.search(re.escape(keyword), text, flags=re.IGNORECASE)
        if match:
            normalized.append(match.group(0).strip())

    return unique_strings(normalized)


def build_radar_metrics(match_score: int) -> list[RadarMetric]:
    metric_names = ["技术深度", "项目经验", "软技能", "背景示例", "AI技能"]
    offsets = [0, -6, -10, -4, -8]
    metrics: list[RadarMetric] = []
    for index, name in enumerate(metric_names):
        value = max(0, min(100, match_score + (offsets[index] if index < len(offsets) else -8)))
        metrics.append(RadarMetric(name=name, value=value))
    return metrics


def chunk_to_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def build_chat_fallback(question: str, context: str) -> str:
    if context:
        snippet = context[:500]
        return (
            "模型流式调用失败，以下是基于已检索简历片段的降级回答："
            f"\n问题：{question}\n简历证据：{snippet}"
        )
    return "暂时无法生成回答，因为没有检索到可用的简历上下文。"


def build_resume_sources(resume_text: str) -> list[dict[str, str]]:
    normalized_sections = [section.strip() for section in re.split(r"\n{2,}", resume_text) if section.strip()]
    if not normalized_sections:
        normalized_sections = [resume_text.strip()]

    if len(normalized_sections) == 1:
        # 如果解析器只给出一整段文本，这里再按常见简历标题和句号做一次兜底切分，
        # 避免所有证据来源都指向同一大段内容。
        normalized_sections = [
            section.strip()
            for section in re.split(
                r"(?=(?:工作经历|项目经历|项目经验|教育经历|教育背景|技能清单|专业技能|个人优势|自我评价|Summary|Experience|Projects|Education|Skills))|(?<=[。；;])\s+",
                resume_text,
            )
            if section.strip()
        ] or normalized_sections

    sources: list[dict[str, str]] = []
    seen_snippets: set[str] = set()
    for index, section in enumerate(normalized_sections[:8], start=1):
        snippet = re.sub(r"\s+", " ", section).strip()
        if not snippet:
            continue
        snippet_key = snippet.lower()
        if snippet_key in seen_snippets:
            continue
        seen_snippets.add(snippet_key)
        sources.append({"source_id": f"resume_{index}", "snippet": snippet[:700]})
    return sources


def format_sources_for_prompt(sources: list[dict[str, str]]) -> str:
    if not sources:
        return "无可用证据片段"
    return "\n\n".join(
        f"[{source['source_id']}] {source['snippet']}" for source in sources
    )


def format_context_docs(context_docs: list[dict[str, str]]) -> str:
    if not context_docs:
        return ""
    return "\n\n".join(
        f"[{doc['source_id']}] {doc['snippet']}" for doc in context_docs
    )


def normalize_sources(raw_sources: Any, fallback_sources: list[dict[str, str]]) -> list[dict[str, str]]:
    source_candidates = raw_sources if isinstance(raw_sources, list) and raw_sources else fallback_sources
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(source_candidates, start=1):
        item = candidate if isinstance(candidate, dict) else {}
        source_id = str(item.get("source_id") or f"source_{index}").strip()
        snippet = str(item.get("snippet") or "").strip()
        if not source_id or not snippet or source_id in seen:
            continue
        seen.add(source_id)
        normalized.append(EvidenceSource(source_id=source_id, snippet=snippet).model_dump())
    return normalized


def normalize_source_ids(raw_source_ids: Any, sources: list[dict[str, str]]) -> list[str]:
    valid_source_ids = {source["source_id"] for source in sources}
    if not isinstance(raw_source_ids, list):
        return []
    normalized: list[str] = []
    for item in raw_source_ids:
        source_id = str(item).strip()
        if source_id and source_id in valid_source_ids and source_id not in normalized:
            normalized.append(source_id)
    return normalized


def normalize_evaluation_items(raw_items: Any, sources: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            source_ids = normalize_source_ids(item.get("source_ids"), sources)
        else:
            text = str(item).strip()
            source_ids = []
        if not text:
            continue
        normalized.append(EvaluationItem(text=text, source_ids=source_ids).model_dump())
    return normalized


def fallback_evaluation(
    resume_text: str,
    jd_keywords: Iterable[str],
    sources: list[dict[str, str]],
) -> dict:
    lowered_resume = resume_text.lower()
    keywords = unique_strings(list(jd_keywords))
    matched = [keyword for keyword in keywords if keyword.lower() in lowered_resume]
    coverage = len(matched) / max(len(keywords), 1)
    match_score = max(45, min(95, round(coverage * 100))) if keywords else 60
    title = "高度匹配" if match_score >= 80 else "有一定匹配度" if match_score >= 65 else "建议谨慎评估"
    summary = (
        f"候选人与JD的整体匹配度约为 {match_score} 分。"
        f"当前证据主要来自简历中出现的关键词：{', '.join(matched[:4]) or '暂未识别到明确重合项'}。"
    )
    normalized_sources = normalize_sources(None, sources)
    default_source_ids = [source["source_id"] for source in normalized_sources[:2]]
    highlights = [
        {"text": f"简历中明确出现关键词：{keyword}", "source_ids": default_source_ids}
        for keyword in matched[:4]
    ] or [{"text": "简历已成功解析，可继续结合项目经历人工复核。", "source_ids": default_source_ids}]
    risks = []
    missing = [keyword for keyword in keywords if keyword not in matched]
    if missing:
        risks.append(
            {
                "text": f"以下JD关键词在简历中缺少直接证据：{', '.join(missing[:4])}",
                "source_ids": [],
            }
        )
    risks.append({"text": "当前结果为降级评估，建议结合面试继续确认细节。", "source_ids": default_source_ids})
    radar_metrics = [metric.model_dump() for metric in build_radar_metrics(match_score)]
    return {
        "summary": summary,
        "summary_source_ids": default_source_ids,
        "title": title,
        "decision": title,
        "match_score": match_score,
        "radar_metrics": radar_metrics,
        "highlights": highlights,
        "risks": risks,
        "sources": normalized_sources,
    }
