import re
from docx import Document


class ResumeParser:
    def __init__(self, chunk_size=200, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _clean_text(self, text):
        """清理文本，去除多余的空白、制表符和换行"""
        # 1. 替换制表符
        text = text.replace("\t", " ")

        # 多个连续空格合并为1个
        text = re.sub(r" +", " ", text)

        # 多个连续换行合并为1个
        text = re.sub(r"\n+", "\n", text)
        return text.strip()

    def extract_from_docx(self, file_path):
        """从docx文件中提取文本内容并清洗"""
        doc = Document(file_path)
        content = []

        # 1. 提取段落文本
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # 只添加非空段落
                content.append(paragraph.text)
        # 2. TODO 提取表格文本

        raw_text = "\n".join(content)

        return self._clean_text(raw_text)

    def get_chunks(self, text):
        """将文本分割成指定大小的块，支持重叠"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunks.append(text[start:end])
            # 移动步长 = 窗口大小 - 重叠大小
            start += self.chunk_size - self.overlap

            # 剩余不足一个步长，停止
            if start >= len(text):
                break

        return chunks
