from typing import cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.app_context import agent
from core.route_helpers import ensure_playground_enabled
from database import get_db
from llm_runtime import build_preview
from models import (
    PromptConfig,
    PromptPlaygroundRunRequest,
    PromptPlaygroundRunResponse,
    PromptScenarioResponse,
    PromptVersion,
    PromptVersionCreateRequest,
    PromptVersionResponse,
    UsageMetricsResponse,
)
from agent_prompts import PROMPT_SCENARIOS, get_prompt_scenario


router = APIRouter(prefix="/prompt-playground")


@router.get("/scenarios", summary="Prompt Playground 场景列表")
async def prompt_playground_scenarios():
    ensure_playground_enabled()
    items = [
        PromptScenarioResponse(
            prompt_name=scenario.prompt_name,
            label=scenario.label,
            description=scenario.description,
            output_mode=scenario.output_mode,
            output_schema_name=scenario.output_schema_name,
            default_system_instruction=scenario.default_system_instruction,
            default_user_template=scenario.default_user_template,
            default_config=scenario.default_config,
            fields=scenario.fields,
        )
        for scenario in PROMPT_SCENARIOS.values()
    ]
    return {"items": [item.model_dump() for item in items]}


@router.get("/versions", summary="Prompt Playground 版本列表")
async def prompt_playground_versions(prompt_name: str, db: AsyncSession = Depends(get_db)):
    ensure_playground_enabled()
    stmt = select(PromptVersion).where(PromptVersion.prompt_name == prompt_name).order_by(desc(PromptVersion.created_at))
    versions = (await db.execute(stmt)).scalars().all()
    items = [
        PromptVersionResponse(
            id=version.id,
            prompt_name=version.prompt_name,
            version_label=version.version_label,
            system_instruction=version.system_instruction,
            user_template=version.user_template,
            config=PromptConfig(**(version.config_json or {})),
            note=version.note,
            created_at=version.created_at.isoformat(),
        )
        for version in versions
    ]
    return {"items": [item.model_dump() for item in items]}


@router.post("/versions", summary="保存 Prompt Playground 版本")
async def prompt_playground_create_version(payload: PromptVersionCreateRequest, db: AsyncSession = Depends(get_db)):
    ensure_playground_enabled()
    get_prompt_scenario(payload.prompt_name)
    version = PromptVersion(
        prompt_name=payload.prompt_name,
        version_label=payload.version_label,
        system_instruction=payload.system_instruction,
        user_template=payload.user_template,
        config_json=payload.config.model_dump(),
        note=payload.note,
    )
    db.add(version)
    await db.flush()
    return PromptVersionResponse(
        id=version.id,
        prompt_name=version.prompt_name,
        version_label=version.version_label,
        system_instruction=version.system_instruction,
        user_template=version.user_template,
        config=payload.config,
        note=version.note,
        created_at=version.created_at.isoformat(),
    ).model_dump()


@router.post("/run", summary="运行 Prompt Playground")
async def prompt_playground_run(payload: PromptPlaygroundRunRequest, db: AsyncSession = Depends(get_db)):
    ensure_playground_enabled()
    scenario = get_prompt_scenario(payload.prompt_name)
    prompt_version = None
    if payload.prompt_version_id is not None:
        prompt_version = (
            await db.execute(select(PromptVersion).where(PromptVersion.id == payload.prompt_version_id))
        ).scalar_one_or_none()
        if prompt_version is None:
            raise HTTPException(status_code=404, detail="未找到对应的 prompt 版本")

    system_instruction = payload.system_instruction_override
    user_template = payload.user_template_override
    config = scenario.default_config.model_copy(deep=True)
    if prompt_version is not None:
        system_instruction = system_instruction or prompt_version.system_instruction
        user_template = user_template or prompt_version.user_template
        config = PromptConfig(**(prompt_version.config_json or {}))
    if system_instruction is None:
        system_instruction = scenario.default_system_instruction
    if user_template is None:
        user_template = scenario.default_user_template
    if payload.model_name is not None:
        config.model_name = payload.model_name
    if payload.temperature is not None:
        config.temperature = payload.temperature
    if payload.top_p is not None:
        config.top_p = payload.top_p
    if payload.max_tokens is not None:
        config.max_tokens = payload.max_tokens

    prompt = scenario.builder(system_instruction)
    if user_template != scenario.default_user_template:
        prompt_template = cast(object, prompt.messages[-1])
        message_prompt = getattr(prompt_template, "prompt", None)
        if message_prompt is None or not hasattr(message_prompt, "template"):
            raise HTTPException(status_code=400, detail="当前 prompt 场景不支持覆盖 user template")
        message_prompt.template = user_template

    llm = agent._build_llm(
        model_name=config.model_name,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
    )
    run = await agent._invoke_structured(
        db=db,
        prompt=prompt,
        schema=scenario.schema_type or dict,
        payload=payload.variables,
        model_name=config.model_name,
        source="playground",
        feature="prompt_playground",
        stage=payload.prompt_name,
        prompt_name=payload.prompt_name,
        prompt_version_id=prompt_version.id if prompt_version else None,
        llm=llm,
        extra_json={"save_log": payload.save_log},
    )
    return PromptPlaygroundRunResponse(
        request_id=run.request_id,
        log_id=run.log_id,
        resolved_prompt={
            "system_instruction": system_instruction,
            "user_template": user_template,
        },
        parsed_output=run.parsed_output,
        raw_output_preview=build_preview(run.raw_output),
        usage=UsageMetricsResponse(**run.usage),
        latency_ms=run.latency_ms,
        estimated_cost=run.estimated_cost,
        success=run.success,
        error_message=run.error_message,
    ).model_dump()
