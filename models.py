from datetime import datetime
import enum
import uuid
from typing import Any, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class RadarMetric(BaseModel):
    name: str
    value: int = Field(..., ge=0, le=100)
    max: int = Field(default=100, ge=1)


class EvidenceSource(BaseModel):
    source_id: str
    snippet: str


class EvaluationItem(BaseModel):
    text: str
    source_ids: List[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    summary: str
    summary_source_ids: List[str] = Field(default_factory=list)
    title: str
    decision: str
    match_score: int = Field(..., ge=0, le=100)
    radar_metrics: List[RadarMetric] = Field(default_factory=list)
    highlights: List[EvaluationItem] = Field(default_factory=list)
    risks: List[EvaluationItem] = Field(default_factory=list)
    sources: List[EvidenceSource] = Field(default_factory=list)


class EvaluationScoreResult(BaseModel):
    title: str
    decision: str
    match_score: int = Field(..., ge=0, le=100)


class EvaluationSummaryResult(BaseModel):
    summary: str
    summary_source_ids: List[str] = Field(default_factory=list)


class EvaluationItemsResult(BaseModel):
    items: List[EvaluationItem] = Field(default_factory=list)


class ResumeStatus(enum.Enum):
    PENDING = "pending"
    PARSING = "parsing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(50), nullable=False)
    candidate_name: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    phone: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[ResumeStatus] = mapped_column(
        Enum(ResumeStatus), default=ResumeStatus.PENDING, nullable=False
    )
    content: Mapped[Optional[str]] = mapped_column(Text)
    evaluation_result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )

    __table_args__ = (
        UniqueConstraint("user_id", "phone", name="_user_phone_uc"),
        Index("ix_user_phone", "user_id", "phone"),
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    resume_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    candidate_name: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    interview_identity: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="draft", nullable=False)
    questions: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    answers: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt_name: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    version_label: Mapped[str] = mapped_column(String(80), nullable=False)
    system_instruction: Mapped[str] = mapped_column(Text, nullable=False)
    user_template: Mapped[str] = mapped_column(Text, nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)

    __table_args__ = (
        Index("ix_prompt_versions_name_created", "prompt_name", "created_at"),
    )


class ModelCallLog(Base):
    __tablename__ = "model_call_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    source: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    feature: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    stage: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    prompt_name: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    prompt_version_id: Mapped[Optional[int]] = mapped_column(Integer, index=True, nullable=True)
    input_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    output_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extra_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)

    __table_args__ = (
        Index("ix_model_call_logs_created", "created_at"),
        Index("ix_model_call_logs_source_created", "source", "created_at"),
        Index("ix_model_call_logs_feature_created", "feature", "created_at"),
        Index("ix_model_call_logs_prompt_created", "prompt_name", "created_at"),
    )


class QueryRequest(BaseModel):
    user_id: str
    text: str
    candidate_name: Optional[str] = None
    resume_id: Optional[int] = None


class EvaluationRequest(BaseModel):
    user_id: str
    jd_text: str
    resume_id: Optional[int] = None
    candidate_name: Optional[str] = None
    phone: Optional[str] = None
    jd_keywords: Optional[List[str]] = None


class JDAnalysisRequest(BaseModel):
    jd_text: str


class JDAnalysisResponse(BaseModel):
    keywords: List[str] = Field(default_factory=list)


class JDKeywordExtractionResult(BaseModel):
    keywords: List[str] = Field(default_factory=list, description="仅包含来自JD原文的关键词")


class ChatRequest(BaseModel):
    user_id: str
    text: str
    role: str = "user"
    candidate_name: Optional[str] = None
    resume_id: Optional[int] = None


class OCRResponse(BaseModel):
    text: str


class ChatSuggestionsResponse(BaseModel):
    suggestions: List[str] = Field(default_factory=list)


class InterviewQuestion(BaseModel):
    question_id: str
    category: str
    question: str
    intent: str
    source_ids: List[str] = Field(default_factory=list)


class InterviewQuestionsResponse(BaseModel):
    questions: List[InterviewQuestion] = Field(default_factory=list)


class InterviewQuestionResult(BaseModel):
    question_id: str
    score: int = Field(..., ge=0, le=100)
    feedback: str
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)


class InterviewEvaluationLLMResult(BaseModel):
    overall_feedback: str
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    question_results: List[InterviewQuestionResult] = Field(default_factory=list)


class InterviewAnswerInput(BaseModel):
    question_id: str
    question: str
    category: str
    answer: str


class InterviewStartRequest(BaseModel):
    user_id: str
    jd_text: str
    interview_identity: str
    resume_id: Optional[int] = None
    candidate_name: Optional[str] = None
    phone: Optional[str] = None
    jd_keywords: Optional[List[str]] = None


class InterviewSubmitRequest(BaseModel):
    user_id: str
    jd_text: str
    interview_identity: str
    session_id: Optional[str] = None
    answers: List[InterviewAnswerInput] = Field(default_factory=list)
    resume_id: Optional[int] = None
    candidate_name: Optional[str] = None
    phone: Optional[str] = None
    jd_keywords: Optional[List[str]] = None


class InterviewSubmitResult(BaseModel):
    total_score: int = Field(..., ge=0, le=100)
    verdict: str
    overall_feedback: str
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    question_results: List[InterviewQuestionResult] = Field(default_factory=list)


class InterviewHistoryRequest(BaseModel):
    user_id: str
    interview_identity: str
    resume_id: Optional[int] = None


class InterviewHistoryItem(BaseModel):
    session_id: str
    interview_identity: str
    candidate_name: str
    verdict: str
    total_score: int = Field(..., ge=0, le=100)
    created_at: str


class InterviewHistoryResponse(BaseModel):
    items: List[InterviewHistoryItem] = Field(default_factory=list)


class InterviewSessionDetailResponse(BaseModel):
    session_id: str
    interview_identity: str
    candidate_name: str
    status: str
    questions: List[dict[str, Any]] = Field(default_factory=list)
    answers: List[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class PromptConfig(BaseModel):
    model_name: str
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)


class PromptVersionCreateRequest(BaseModel):
    prompt_name: str
    version_label: str
    system_instruction: str
    user_template: str
    config: PromptConfig
    note: Optional[str] = None


class PromptVersionResponse(BaseModel):
    id: int
    prompt_name: str
    version_label: str
    system_instruction: str
    user_template: str
    config: PromptConfig
    note: Optional[str] = None
    created_at: str


class PromptScenarioField(BaseModel):
    name: str
    label: str
    description: str
    multiline: bool = False


class PromptScenarioResponse(BaseModel):
    prompt_name: str
    label: str
    description: str
    output_mode: str
    output_schema_name: Optional[str] = None
    default_system_instruction: str
    default_user_template: str
    default_config: PromptConfig
    fields: List[PromptScenarioField] = Field(default_factory=list)


class PromptPlaygroundRunRequest(BaseModel):
    prompt_name: str
    prompt_version_id: Optional[int] = None
    system_instruction_override: Optional[str] = None
    user_template_override: Optional[str] = None
    variables: dict[str, Any] = Field(default_factory=dict)
    model_name: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    save_log: bool = True


class UsageMetricsResponse(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class PromptPlaygroundRunResponse(BaseModel):
    request_id: str
    log_id: Optional[int] = None
    resolved_prompt: dict[str, str]
    parsed_output: Any = None
    raw_output_preview: str = ""
    usage: UsageMetricsResponse = Field(default_factory=UsageMetricsResponse)
    latency_ms: int
    estimated_cost: Optional[float] = None
    success: bool
    error_message: Optional[str] = None


class ObservabilitySummaryResponse(BaseModel):
    total_calls: int = 0
    success_rate: float = 0
    fallback_rate: float = 0
    avg_latency_ms: float = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_estimated_cost: float = 0


class ObservabilityLogItemResponse(BaseModel):
    id: int
    request_id: str
    source: str
    feature: str
    stage: str
    model_name: str
    prompt_name: str
    prompt_version_id: Optional[int] = None
    input_summary: str = ""
    output_summary: str = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: int
    estimated_cost: Optional[float] = None
    success: bool
    fallback_used: bool
    error_message: Optional[str] = None
    created_at: str


class ObservabilityLogsResponse(BaseModel):
    items: List[ObservabilityLogItemResponse] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


class ObservabilityTrendPoint(BaseModel):
    bucket: str
    latency_ms_avg: float = 0
    total_tokens: int = 0
    total_estimated_cost: float = 0
    total_calls: int = 0


class ObservabilityTrendsResponse(BaseModel):
    points: List[ObservabilityTrendPoint] = Field(default_factory=list)
