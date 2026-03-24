from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from agent import ResumeAgent
from contextlib import asynccontextmanager
import os
import shutil


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时连接数据库
    await agent.init_db()
    yield
    # 关闭时断开数据库连接
    await agent.engine.dispose()
    await agent.redis_client.close()


app = FastAPI(lifespan=lifespan)
agent = ResumeAgent()


class QueryRequest(BaseModel):
    user_id: str  # 标识用户
    text: str


@app.get("/")
async def root():
    return {"message": "AI职业经纪人已启动，访问/chat接口进行对话"}


@app.post(
    "/upload_resume",
    summary="上传简历文件",
    description="上传docx格式的简历文件，AI经纪人将解析内容并存储",
)
async def upload_resume(file: UploadFile = File(...), candidate_name: str = Form(...)):
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="仅支持docx格式的文件")

    # 1. 临时保存文件
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. 调用Agent方法处理文件
        num_chunks = agent.add_resume(temp_path, candidate_name)
        return {
            "status": "success",
            "candidate_name": candidate_name,
            "chunks_added": num_chunks,
            "message": "简历已解析并存入向量库",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 3. 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/query", summary="查询接口", description="发送查询文本，AI经纪人返回回答")
async def query_resume(
    question: str = Form(...),
    user_id: str = Form(...),
    candidate_name: str = Form(None),
):
    # 构建过滤条件
    search_filter = None
    if candidate_name:
        search_filter = {"candidate": candidate_name}
    answer = await agent.ask(question, user_id, search_filter)
    return {"reply": answer}


@app.post(
    "/chat",
    summary="与AI职业经纪人对话",
    description="发送问题给 AI 经纪人，支持多轮对话记忆",
)
async def chat_endpoint(request: QueryRequest, background_tasks: BackgroundTasks):
    answer = await agent.ask(request.text, request.user_id)
    background_tasks.add_task(agent.async_persistence, request.user_id, answer)
    return {"reply": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
