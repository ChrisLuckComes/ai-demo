from typing import List


"""
将长文本切分为固定大小的块，并保留重叠部分
chunk_size: 每个块的最大字符数
overlap: 块之间重叠的字符数，确保上下文连续性
"""


def simple_chunker(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # 步进 = 大小 - 重叠
        start += chunk_size - overlap
    return chunks


# 示例用法
if __name__ == "__main__":
    long_text = "这是一个很长的文本，需要被切分成多个块。每个块的大小是固定的，并且块之间有重叠部分，以确保上下文的连续性。这个函数可以帮助我们处理长文本，使其适合输入到语言模型中。"
    chunks = simple_chunker(long_text, chunk_size=20, overlap=5)
    print(f"切分成了{len(chunks)}个块")
