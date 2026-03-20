from pydantic import BaseModel
from typing import List

class AIResponse(BaseModel):
    content: str
    model_name: str

class VectorData(BaseModel):
    text: str
    embedding: List[float] # 向量，一个浮点数列表