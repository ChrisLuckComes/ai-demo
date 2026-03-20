# def format_prompt(role:str, content:str) -> str:
#     return f"[{role.upper()}]:{content}"

messages = [
    {"role": "system", "content": "你是一个资深前端架构师"},
    {"role": "user", "content": "如何学习Python?"},
    {"role": "assistant", "content": "建议从语法映射开始"},
]


if __name__ == "__main__":
    # for msg in messages:
    #     role = msg["role"].upper()
    #     content = msg["content"]
    #     print(f"{role}: {content}")

    # 使用列表推导式转换
    # contents = [msg["content"] for msg in messages]

    # 使用列表推导式转换+过滤
    # userContent = [msg["content"] for msg in messages if msg["role"] == "user"]

    search_results = [
        {"score": 0.95, "content": "Python是一种流行的编程语言，适合初学者学习。"},
        {
            "score": 0.42,
            "content": "今天天气不错",
        },
        {
            "score": 0.88,
            "content": "RAG架构可以有效解决大模型的幻觉问题",
        },
    ]

    context_list = [
        result["content"] for result in search_results if result["score"] > 0.8
    ]

    print(context_list)
