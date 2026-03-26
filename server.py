import json

from fastapi import (
    FastAPI,
    BackgroundTasks,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    Request,
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from models import JDAnalysisRequest, QueryRequest, EvaluationRequest, Resume, ResumeStatus
from agent import ResumeAgent
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from database import engine, get_db, redis_client, Base
import os
import shutil
import hashlib


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时连接数据库
    # 1. 【启动时】初始化表结构 (替代原来的 agent.init_db)
    async with engine.begin() as conn:
        # 这行代码会检查 models.py 中定义的 Base 类，并创建所有不存在的表
        await conn.run_sync(Base.metadata.create_all)

    print("数据库连接已建立，表结构已初始化")
    yield
    # 关闭时断开数据库连接
    await agent.engine.dispose()
    await agent.redis_client.close()
    print("数据库和Redis连接已断开")


app = FastAPI(lifespan=lifespan)
agent = ResumeAgent(redis_client)

# CORS 配置，允许所有来源（可根据需要指定 origins）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "AI职业经纪人已启动，访问/chat接口进行对话"}


@app.post("/analyze_jd", summary="分析JD描述，返回关键词")
async def analyze_jd(request: JDAnalysisRequest):
    jd_text = request.jd_text.strip()
    if not jd_text:
        return {"keywords": []}
    # 生成JD摘要作为缓存key
    jd_hash = hashlib.md5(jd_text.encode("utf-8")).hexdigest()
    cache_key = f"jd_analysis:{jd_hash}"
    # 检查redis缓存
    if agent.redis_client:
        cached = await agent.redis_client.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass
    # AI分析JD关键词
    keywords = await agent.analyze_jd_keywords(jd_text)
    result = {"jd": jd_text, "keywords": keywords}
    # 存入redis缓存，有效期60分钟
    if agent.redis_client:
        await agent.redis_client.setex(cache_key, 3600, json.dumps(result, ensure_ascii=False))
    return result


@app.post(
    "/upload_resume",
    summary="上传简历文件",
    description="上传docx格式的简历文件，AI经纪人将解析内容并存储",
)
async def upload_resume(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    candidate_name: str = Form(...),
    phone: str = Form(...),
    user_id: str = Form(...),
):
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="仅支持docx格式的文件")

    # 先查询是否存在记录
    stmt = select(Resume).where(Resume.user_id == user_id, Resume.phone == phone)
    result = await db.execute(stmt)
    existing_resume = result.scalar_one_or_none()

    if existing_resume:
        # 如果记录存在，更新状态为PENDING，等待重新处理
        existing_resume.status = ResumeStatus.PENDING
        new_resume = existing_resume
    else:
        # 数据库创建初始记录
        new_resume = Resume(
            user_id=user_id,
            candidate_name=candidate_name,
            phone=phone,
            status=ResumeStatus.PENDING,  # 初始状态为解析中
        )
        db.add(new_resume)
    await db.commit()
    await db.refresh(new_resume)

    background_tasks.add_task(
        agent.handle_resume_process,
        resume_id=new_resume.id,
        file_content=await file.read(),
        candidate_name=candidate_name,
        phone=phone,
    )

    return {
        "message": "简历上传成功，正在后台处理",
        "resume_id": new_resume.id,
        "status": "pending",
    }

@app.post(
    "/evaluate",
    summary="评估简历",
    description="根据候选人姓名和简历内容，AI经纪人返回评估结果",
)
async def evaluate_resume(request: EvaluationRequest, db: Session = Depends(get_db)):
    # 1. 校验参数，JD和JD关键词必填
    jd = getattr(request, "jd", None)
    jd_keywords = getattr(request, "jd_keywords", None)
    if not jd or not jd_keywords:
        raise HTTPException(status_code=400, detail="请先进行JD分析，获取JD描述和关键词")

    # 2. 查询数据库有没有这个简历
    existing_resume = await agent.check_and_get_evaluation(
        request.user_id, request.candidate_name, request.phone, db=db
    )
    if not existing_resume:
        raise HTTPException(
            status_code=404,
            detail="未找到对应的简历，请先上传简历后再进行评估",
        )

    # 若数据库已有结构化评估结果且包含雷达图数据，直接返回
    if existing_resume.evaluation_result:
        result = existing_resume.evaluation_result
        if "radarData" in result and "radarIndicators" in result:
            return {"evaluation": result}
        if "radar_scores" in result and "radar_indicators" in result:
            result["radarData"] = result["radar_scores"]
            result["radarIndicators"] = result["radar_indicators"]
            return {"evaluation": result}

    # 3. 评估时传递JD内容和关键词
    evaluation = await agent.evaluate_resume(existing_resume.content, jd=jd, jd_keywords=jd_keywords)
    return {"evaluation": evaluation}


@app.post("/query", summary="查询接口", description="发送查询文本，AI经纪人返回回答")
async def query_resume(
    question: str = Form(...),
    user_id: str = Form(...),
    candidate_name: str = Form(None),
    db: Session = Depends(get_db),
):
    # 构建过滤条件
    search_filter = None
    if candidate_name:
        search_filter = {"candidate": candidate_name}
    answer = await agent.ask(question, user_id, search_filter, db=db)
    return {"reply": answer}


# SSE流式响应的/chat接口
@app.post(
    "/chat",
    summary="与AI职业经纪人对话（SSE流式）",
    description="发送问题给 AI 经纪人，流式返回回复内容，支持多轮对话记忆",
)
async def chat_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    req_json = await request.json()
    user_id = req_json.get("user_id")
    text = req_json.get("text")
    # 获取完整AI回复
    answer = agent.ask(text, user_id, db=db)

    return StreamingResponse(answer, media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
