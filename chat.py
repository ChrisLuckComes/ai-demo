import os
import asyncio
from google import genai
from dotenv import load_dotenv
from models import AIResponse

# 加载环境变量
load_dotenv()

# 配置API KEY
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = "gemini-3-flash-preview"


async def ask_gemini(prompt: str):
    # 发送请求
    response = await client.aio.models.generate_content(model=model, contents=prompt)

    return AIResponse(content=response.text, model_name=model)


async def main():
    answer = await ask_gemini("江波龙今天公告减持，请分析一下这个消息对江波龙的影响？")
    print(f"AI回答: {answer.content} (模型: {answer.model_name})")


if __name__ == "__main__":
    asyncio.run(main())
