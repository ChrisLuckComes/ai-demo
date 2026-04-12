import inspect

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import Base, engine, redis_client
from routes.interview import router as interview_router
from routes.observability import router as observability_router
from routes.playground import router as playground_router
from routes.resume import router as resume_router


async def _close_redis() -> None:
    if not redis_client:
        return
    close_method = getattr(redis_client, "aclose", None) or getattr(redis_client, "close", None)
    if close_method is None:
        return
    result = close_method()
    if inspect.isawaitable(result):
        await result


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()
    await _close_redis()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "resume-agent backend is running",
        "endpoints": [
            "/analyze_jd",
            "/ocr_jd_image",
            "/upload_resume",
            "/resumes",
            "/evaluate",
            "/chat",
        ],
    }


app.include_router(resume_router)
app.include_router(interview_router)
app.include_router(observability_router)
app.include_router(playground_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
