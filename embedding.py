import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text: str):
    # 调用gemini的嵌入模型
    result = client.models.embed_content(model="gemini-embedding-001", contents=text)

    return result.embeddings[0].values

if __name__ == "__main__":
    text = "这是一个测试文本，用于生成嵌入向量。"
    embedding_vector = get_embedding(text)
    print(f"文本: {text}")
    print(f"向量维度: {len(embedding_vector)}") # 向量维度
    print(f"前5位数据: {embedding_vector[:5]}...")  # 只显示前5个维度的值