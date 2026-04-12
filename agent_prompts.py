from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from models import (
    ChatSuggestionsResponse,
    EvaluationItemsResult,
    EvaluationScoreResult,
    EvaluationSummaryResult,
    InterviewEvaluationLLMResult,
    InterviewQuestionsResponse,
    JDKeywordExtractionResult,
    PromptConfig,
    PromptScenarioField,
)


SYSTEM_INSTRUCTION = (
    "你是一名专业、冷静、重证据的招聘顾问。"
    "你的判断必须基于提供的简历内容、JD信息和检索片段。"
    "禁止补充候选人未明确写出的经历、技能、业绩或职责，禁止基于常识脑补。"
    "如果证据不足，必须明确说明“证据不足”或“简历中没有直接体现”。"
    "结论要服务招聘决策，语言简洁、直接、可执行，避免空泛表扬。"
    "输出时优先引用具体事实，例如项目、技术、职责、年限、业务场景。"
)


@dataclass(frozen=True)
class PromptScenario:
    prompt_name: str
    label: str
    description: str
    default_system_instruction: str
    default_user_template: str
    default_config: PromptConfig
    fields: list[PromptScenarioField]
    output_mode: str
    output_schema_name: str | None
    schema_type: type | None
    builder: callable


DEFAULT_PROMPT_CONFIG = PromptConfig(
    model_name="gemini-2.5-flash",
    temperature=0,
    top_p=None,
    max_tokens=None,
)


def extract_user_template(prompt: ChatPromptTemplate) -> str:
    for message in reversed(prompt.messages):
        template = getattr(getattr(message, "prompt", None), "template", None)
        if isinstance(template, str):
            return template
    return ""


def build_follow_up_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """
        你是招聘顾问，请根据本轮问答生成 3 条适合继续追问的问题。
        候选人：{candidate_name}
        用户问题：{question}
        AI回答：{answer}

        要求：
        1. 每条都是中文追问句子
        2. 3条问题分别优先覆盖：技术深度、真实职责或ownership、风险确认
        3. 要紧扣当前问答里已经出现的信息，不要泛泛提问
        4. 避免重复，避免空话，避免一次问太多点
        """
    )


def build_jd_keyword_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是招聘JD关键词抽取助手。你的唯一任务是从输入原文中提取关键词。"
                "禁止补充原文未出现的词，禁止改写为同义词，禁止根据常识推断。"
                "只保留文本中能直接定位到的技能、工具、业务领域、职责方向、方法论等关键词。"
                "优先保留对招聘筛选最有区分度的词，过滤'负责'、'沟通能力'这类泛词。"
                "如果原文中出现英文、缩写、大小写混合写法，尽量保留原文写法。"
                "不要输出完整句子，不要输出解释。"
                "关键词数量控制在4到8个，若有效关键词不足则按实际数量返回。"
            ),
            (
                "user",
                "请从下面JD原文中提取关键词，仅返回结构化结果。\n\nJD原文：\n{text}",
            ),
        ]
    )


def build_evaluation_prompt(system_instruction: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_instruction
                + " 输出结构必须适合招聘评审，评分范围为0到100。"
                + " 亮点和风险都必须基于简历中的明确证据，禁止写空泛结论。",
            ),
            (
                "user",
                """
                请结合以下信息评估候选人：
                JD关键词: {keywords}

                JD全文:
                {jd_text}

                简历全文:
                {resume_text}

                可引用证据片段:
                {source_block}

                评估原则：
                1. 只能依据简历中明确出现的信息判断。
                2. 如果JD某项要求在简历中没有直接证据，必须视为证据不足，不能脑补。
                3. 亮点必须具体到技能、项目、职责、结果或业务场景，并引用 `source_ids`。
                4. 风险必须具体指出缺口、不确定项或需要面试确认的点，并引用 `source_ids`。
                5. 不要使用“潜力不错”“综合素质较强”这类空泛表述。
                6. `summary_source_ids`、`highlights[].source_ids`、`risks[].source_ids` 只能引用上面给出的 source_id。
                7. 如果某条结论没有直接证据支持，请使用空数组，不要伪造 source_id。

                评分参考：
                - 90到100：核心技能、项目复杂度、业务场景均高度匹配，且证据充分
                - 75到89：大部分要求匹配，存在少量缺口但整体可推进
                - 60到74：有相关经历，但关键要求证据不足或存在明显短板
                - 0到59：核心要求不匹配，或缺少支持结论的直接证据

                请返回：
                - summary: 2到3句中文结论
                - summary_source_ids: 支撑 summary 的 source_id 列表
                - title: 一句简短标题
                - decision: 明确结论
                - match_score: 0到100整数
                - radar_metrics: 4到6个维度，每个维度包含name、value、max
                - highlights: 3到5条亮点，每条包含 text、source_ids
                - risks: 2到4条风险，每条包含 text、source_ids
                - sources: 原样返回上面提供的证据片段，不要新增或改写 source_id
                """,
            ),
        ]
    )


def build_evaluation_score_prompt(system_instruction: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_instruction
                + " 你当前只负责产出评分结论，不要提前输出总结、亮点或风险。",
            ),
            (
                "user",
                """
                请根据以下 JD 与简历内容，输出候选人的匹配评分。

                JD关键词: {keywords}

                JD全文:
                {jd_text}

                简历全文:
                {resume_text}

                评分原则：
                1. 只能基于简历中明确出现的信息打分。
                2. 如果 JD 关键要求缺少直接证据，必须降低分数。
                3. 输出简短标题和明确结论，适合招聘评审页面展示。

                请返回：
                - title
                - decision
                - match_score
                """,
            ),
        ]
    )


def build_evaluation_summary_prompt(system_instruction: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_instruction
                + " 你当前只负责生成招聘总结，并为总结绑定合法 source_id。",
            ),
            (
                "user",
                """
                请基于以下信息，生成 2 到 3 句中文评估总结。

                当前评分：{match_score}
                当前结论：{decision}
                JD关键词: {keywords}

                JD全文:
                {jd_text}

                简历全文:
                {resume_text}

                可引用证据片段:
                {source_block}

                要求：
                1. 只依据简历和 JD 内容，不要补充不存在的信息。
                2. 总结要适合招聘决策，避免空话。
                3. `summary_source_ids` 只能使用给定的 source_id。

                请返回：
                - summary
                - summary_source_ids
                """,
            ),
        ]
    )


def build_evaluation_items_prompt(system_instruction: str, item_type: str) -> ChatPromptTemplate:
    focus_instruction = (
        "你当前只负责生成亮点，每条都要具体，并绑定合法 source_id。"
        if item_type == "highlights"
        else "你当前只负责生成风险点，每条都要具体，并绑定合法 source_id。"
    )
    item_label = "亮点" if item_type == "highlights" else "风险点"
    count_hint = "3到5条" if item_type == "highlights" else "2到4条"
    extra_rule = (
        "亮点必须具体到技能、项目、职责、结果或业务场景。"
        if item_type == "highlights"
        else "风险要具体指出缺口、不确定项或需要面试确认的点。"
    )

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_instruction + f" {focus_instruction}",
            ),
            (
                "user",
                f"""
                请基于以下信息，生成候选人的{item_label}。

                当前评分：{{match_score}}
                当前结论：{{decision}}
                当前总结：{{summary}}
                JD关键词: {{keywords}}

                JD全文:
                {{jd_text}}

                简历全文:
                {{resume_text}}

                可引用证据片段:
                {{source_block}}

                要求：
                1. 只能依据简历和 JD 内容，不要脑补。
                2. {extra_rule}
                3. 每条都必须包含 `text` 和 `source_ids`。
                4. `source_ids` 只能使用给定的 source_id。
                5. 如果没有直接证据支撑，可以使用空数组，但不能伪造来源。

                请返回：
                - items: {count_hint}
                """,
            ),
        ]
    )


def build_chat_prompt(system_instruction: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_instruction
                + " 如果检索内容不足以支撑结论，就明确说明证据不足。"
                + " 优先回答结论，再补充证据；如果用户问是否匹配、是否具备某能力，"
                + "要先给判断，再给依据，最后指出需要进一步确认的风险。",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "相关简历片段:\n{context}\n\n"
                "回答要求:\n"
                "1. 只基于相关简历片段回答。\n"
                "2. 如果问题是判断类问题，优先按“结论 -> 证据 -> 风险/待确认点”作答。\n"
                "3. 如果问题是信息查询类问题，直接列出找到的事实。\n"
                "4. 不要假装看到了上下文之外的信息。\n\n"
                "用户问题:\n{question}",
            ),
        ]
    )


def build_interview_questions_prompt(system_instruction: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_instruction
                + " 你当前只负责生成模拟面试问题。"
                + " 问题必须贴近真实中文面试场景，不要输出答案，不要输出多余解释。",
            ),
            (
                "user",
                """
                请基于以下信息，为候选人生成 10 道结构化模拟面试问题。

                候选人：{candidate_name}
                当前匹配分：{match_score}
                当前评估结论：{decision}
                JD关键词: {keywords}

                JD全文:
                {jd_text}

                简历全文:
                {resume_text}

                可引用证据片段:
                {source_block}

                当前亮点：
                {highlights_text}

                当前风险：
                {risks_text}

                生成要求：
                1. 总共输出 10 道题。
                2. 题型分布必须为：3题技术深挖、2题项目经历或ownership、2题业务场景或问题解决、2题开放性或沟通协作、1题风险确认或压力问题。
                3. 问法必须像真实面试官，不要变成考试题库或概念解释题。
                4. 问题要优先围绕简历中出现过的技术、项目、职责、结果和风险点。
                5. 如果某项 JD 很关键但简历证据不足，应生成验证型问题，而不是假设候选人一定做过。
                6. 每题必须包含：question_id、category、question、intent、source_ids。
                7. `source_ids` 只能使用给定的 source_id；如果无直接对应证据可用空数组。
                8. `question_id` 使用 q1 到 q10。
                9. category 仅可使用：technical_depth、ownership、problem_solving、communication、risk_check。
                """,
            ),
        ]
    )


def build_interview_submit_prompt(system_instruction: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_instruction
                + " 你当前只负责评估模拟面试回答。"
                + " 评分要看回答是否合理、贴题、有逻辑、有细节，不要求与标准答案逐字一致。"
                + " 不要使用机械对答案的方式评分。",
            ),
            (
                "user",
                """
                请基于以下信息，对候选人的模拟面试整场回答进行评分。

                候选人：{candidate_name}
                JD关键词: {keywords}

                JD全文:
                {jd_text}

                简历全文:
                {resume_text}

                当前评估结论：{decision}
                当前匹配分：{match_score}

                面试问答如下：
                {qa_block}

                评分要求：
                1. 不要求与标准答案逐字一致，只要回答言之有理、切题、贴合 JD 与简历背景即可给分。
                2. 回答空泛、跑题、明显缺乏细节、与简历自相矛盾时，应降低分数。
                3. 每题输出 0 到 100 的整数分。
                4. 每题都要给出简洁 feedback，并列出 strengths、improvements。
                5. 需要给出整场 overall_feedback、strengths、risks。
                6. 不要输出 verdict，verdict 由后端根据总分计算。

                输出要求：
                - question_results: 与输入题目一一对应
                - overall_feedback
                - strengths
                - risks
                """,
            ),
        ]
    )


PROMPT_SCENARIOS: dict[str, PromptScenario] = {
    "jd_keyword": PromptScenario(
        prompt_name="jd_keyword",
        label="JD关键词提取",
        description="从 JD 原文中提取结构化关键词。",
        default_system_instruction="你是招聘JD关键词抽取助手。你的唯一任务是从输入原文中提取关键词。",
        default_user_template=extract_user_template(build_jd_keyword_prompt()),
        default_config=PromptConfig(model_name="gemini-2.5-flash", temperature=0.2, top_p=None, max_tokens=None),
        fields=[PromptScenarioField(name="text", label="JD原文", description="原始 JD 内容。", multiline=True)],
        output_mode="structured",
        output_schema_name="JDKeywordExtractionResult",
        schema_type=JDKeywordExtractionResult,
        builder=lambda system_instruction: ChatPromptTemplate.from_messages(
            [
                ("system", system_instruction),
                ("user", extract_user_template(build_jd_keyword_prompt())),
            ]
        ),
    ),
    "follow_up_suggestions": PromptScenario(
        prompt_name="follow_up_suggestions",
        label="追问问题生成",
        description="根据当前问答生成下一轮追问建议。",
        default_system_instruction="你是招聘顾问，请根据本轮问答生成 3 条适合继续追问的问题。",
        default_user_template=extract_user_template(build_follow_up_prompt()),
        default_config=PromptConfig(model_name="gemini-2.5-flash", temperature=0.2, top_p=None, max_tokens=None),
        fields=[
            PromptScenarioField(name="candidate_name", label="候选人", description="候选人名称。"),
            PromptScenarioField(name="question", label="用户问题", description="当前用户问题。", multiline=True),
            PromptScenarioField(name="answer", label="AI回答", description="上一轮 AI 回答。", multiline=True),
        ],
        output_mode="structured",
        output_schema_name="ChatSuggestionsResponse",
        schema_type=ChatSuggestionsResponse,
        builder=lambda system_instruction: ChatPromptTemplate.from_messages(
            [
                ("system", system_instruction),
                ("user", extract_user_template(build_follow_up_prompt())),
            ]
        ),
    ),
    "evaluation_score": PromptScenario(
        prompt_name="evaluation_score",
        label="评估分数生成",
        description="生成候选人与 JD 的匹配分。",
        default_system_instruction=SYSTEM_INSTRUCTION,
        default_user_template=extract_user_template(build_evaluation_score_prompt(SYSTEM_INSTRUCTION)),
        default_config=DEFAULT_PROMPT_CONFIG,
        fields=[
            PromptScenarioField(name="keywords", label="JD关键词", description="逗号分隔的 JD 关键词。"),
            PromptScenarioField(name="jd_text", label="JD全文", description="完整 JD 内容。", multiline=True),
            PromptScenarioField(name="resume_text", label="简历全文", description="完整简历内容。", multiline=True),
        ],
        output_mode="structured",
        output_schema_name="EvaluationScoreResult",
        schema_type=EvaluationScoreResult,
        builder=build_evaluation_score_prompt,
    ),
    "evaluation_summary": PromptScenario(
        prompt_name="evaluation_summary",
        label="评估总结生成",
        description="生成简历评估总结。",
        default_system_instruction=SYSTEM_INSTRUCTION,
        default_user_template=extract_user_template(build_evaluation_summary_prompt(SYSTEM_INSTRUCTION)),
        default_config=DEFAULT_PROMPT_CONFIG,
        fields=[
            PromptScenarioField(name="match_score", label="匹配分", description="当前评估分数。"),
            PromptScenarioField(name="decision", label="评估结论", description="当前评估结论。"),
            PromptScenarioField(name="keywords", label="JD关键词", description="逗号分隔的 JD 关键词。"),
            PromptScenarioField(name="jd_text", label="JD全文", description="完整 JD 内容。", multiline=True),
            PromptScenarioField(name="resume_text", label="简历全文", description="完整简历内容。", multiline=True),
            PromptScenarioField(name="source_block", label="证据块", description="可直接注入 prompt 的证据文本。", multiline=True),
        ],
        output_mode="structured",
        output_schema_name="EvaluationSummaryResult",
        schema_type=EvaluationSummaryResult,
        builder=build_evaluation_summary_prompt,
    ),
    "evaluation_items_highlights": PromptScenario(
        prompt_name="evaluation_items_highlights",
        label="亮点生成",
        description="生成带来源引用的候选人亮点。",
        default_system_instruction=SYSTEM_INSTRUCTION,
        default_user_template=extract_user_template(build_evaluation_items_prompt(SYSTEM_INSTRUCTION, "highlights")),
        default_config=DEFAULT_PROMPT_CONFIG,
        fields=[
            PromptScenarioField(name="match_score", label="匹配分", description="当前评估分数。"),
            PromptScenarioField(name="decision", label="评估结论", description="当前评估结论。"),
            PromptScenarioField(name="summary", label="评估总结", description="当前总结内容。", multiline=True),
            PromptScenarioField(name="keywords", label="JD关键词", description="逗号分隔的 JD 关键词。"),
            PromptScenarioField(name="jd_text", label="JD全文", description="完整 JD 内容。", multiline=True),
            PromptScenarioField(name="resume_text", label="简历全文", description="完整简历内容。", multiline=True),
            PromptScenarioField(name="source_block", label="证据块", description="可直接注入 prompt 的证据文本。", multiline=True),
        ],
        output_mode="structured",
        output_schema_name="EvaluationItemsResult",
        schema_type=EvaluationItemsResult,
        builder=lambda system_instruction: build_evaluation_items_prompt(system_instruction, "highlights"),
    ),
    "evaluation_items_risks": PromptScenario(
        prompt_name="evaluation_items_risks",
        label="风险生成",
        description="生成带来源引用的候选人风险点。",
        default_system_instruction=SYSTEM_INSTRUCTION,
        default_user_template=extract_user_template(build_evaluation_items_prompt(SYSTEM_INSTRUCTION, "risks")),
        default_config=DEFAULT_PROMPT_CONFIG,
        fields=[
            PromptScenarioField(name="match_score", label="匹配分", description="当前评估分数。"),
            PromptScenarioField(name="decision", label="评估结论", description="当前评估结论。"),
            PromptScenarioField(name="summary", label="评估总结", description="当前总结内容。", multiline=True),
            PromptScenarioField(name="keywords", label="JD关键词", description="逗号分隔的 JD 关键词。"),
            PromptScenarioField(name="jd_text", label="JD全文", description="完整 JD 内容。", multiline=True),
            PromptScenarioField(name="resume_text", label="简历全文", description="完整简历内容。", multiline=True),
            PromptScenarioField(name="source_block", label="证据块", description="可直接注入 prompt 的证据文本。", multiline=True),
        ],
        output_mode="structured",
        output_schema_name="EvaluationItemsResult",
        schema_type=EvaluationItemsResult,
        builder=lambda system_instruction: build_evaluation_items_prompt(system_instruction, "risks"),
    ),
    "interview_questions": PromptScenario(
        prompt_name="interview_questions",
        label="面试问题生成",
        description="生成结构化模拟面试问题。",
        default_system_instruction=SYSTEM_INSTRUCTION,
        default_user_template=extract_user_template(build_interview_questions_prompt(SYSTEM_INSTRUCTION)),
        default_config=DEFAULT_PROMPT_CONFIG,
        fields=[
            PromptScenarioField(name="candidate_name", label="候选人", description="候选人名称。"),
            PromptScenarioField(name="match_score", label="匹配分", description="当前评估分数。"),
            PromptScenarioField(name="decision", label="评估结论", description="当前评估结论。"),
            PromptScenarioField(name="keywords", label="JD关键词", description="逗号分隔的 JD 关键词。"),
            PromptScenarioField(name="jd_text", label="JD全文", description="完整 JD 内容。", multiline=True),
            PromptScenarioField(name="resume_text", label="简历全文", description="完整简历内容。", multiline=True),
            PromptScenarioField(name="source_block", label="证据块", description="可直接注入 prompt 的证据文本。", multiline=True),
            PromptScenarioField(name="highlights_text", label="亮点文本", description="格式化后的亮点文本。", multiline=True),
            PromptScenarioField(name="risks_text", label="风险文本", description="格式化后的风险文本。", multiline=True),
        ],
        output_mode="structured",
        output_schema_name="InterviewQuestionsResponse",
        schema_type=InterviewQuestionsResponse,
        builder=build_interview_questions_prompt,
    ),
    "interview_submit": PromptScenario(
        prompt_name="interview_submit",
        label="面试评分",
        description="对完整模拟面试进行评分。",
        default_system_instruction=SYSTEM_INSTRUCTION,
        default_user_template=extract_user_template(build_interview_submit_prompt(SYSTEM_INSTRUCTION)),
        default_config=DEFAULT_PROMPT_CONFIG,
        fields=[
            PromptScenarioField(name="candidate_name", label="候选人", description="候选人名称。"),
            PromptScenarioField(name="keywords", label="JD关键词", description="逗号分隔的 JD 关键词。"),
            PromptScenarioField(name="jd_text", label="JD全文", description="完整 JD 内容。", multiline=True),
            PromptScenarioField(name="resume_text", label="简历全文", description="完整简历内容。", multiline=True),
            PromptScenarioField(name="decision", label="评估结论", description="当前评估结论。"),
            PromptScenarioField(name="match_score", label="匹配分", description="当前评估分数。"),
            PromptScenarioField(name="qa_block", label="问答块", description="格式化后的面试问答文本。", multiline=True),
        ],
        output_mode="structured",
        output_schema_name="InterviewEvaluationLLMResult",
        schema_type=InterviewEvaluationLLMResult,
        builder=build_interview_submit_prompt,
    ),
}


def get_prompt_scenario(prompt_name: str) -> PromptScenario:
    scenario = PROMPT_SCENARIOS.get(prompt_name)
    if scenario is None:
        raise KeyError(prompt_name)
    return scenario
