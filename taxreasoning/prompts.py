"""All prompt templates used by TaxReasoning.

The strings here are reproduced verbatim from the scripts used for the paper
(archived under ``legacy/``) so that results stay comparable. Do not reword
them without re-running the affected experiments.
"""

# --- Answer generation -------------------------------------------------------

SYSTEM_COT = (
    "你作为资深税务会计师，需解决以下税法问题。"
    "首先，您需要逐步分析问题，并记录每个必要步骤。"
    "最后，您需要在最后一句中总结您的回答，"
    "并以“因此，答案是{最终答案}”结尾。"
    "最终答案应为一个数值。"
)

SYSTEM_PAL = (
    "作为资深税务会计师，您需要生成Python程序解决税法问题。请遵循以下要求：\n\n"
    "1. 程序必须包含在函数 solution() 中，无参数，直接返回最终答案\n"
    "2. 变量命名：使用有税务含义的英文名（如gross_income），避免数字开头的名称\n"
    "3. 计算步骤：\n"
    "   a) 明确定义所有输入变量（硬编码在代码中）\n"
    "   b) 分步注释计算逻辑\n"
    "   c) 最终返回计算结果（不要打印）\n"
    "4. 错误预防：避免除零错误，确保单位统一（如货币单位）\n\n"
    "示例格式：\n"
    "'''python\n"
    "def solution():\n"
    "    # 定义税率和收入\n"
    "    tax_rate = 0.25  # 25%税率\n"
    "    taxable_income = 50000  # 应税收入\n"
    "    \n"
    "    # 计算应纳税额\n"
    "    tax_payable = taxable_income * tax_rate\n"
    "    \n"
    "    return tax_payable  # 直接返回结果\n"
    "'''\n\n"
    "现在请解决以下问题："
)


def user_cot(question: str) -> str:
    """Closed-book chain-of-thought: no supporting regulation text."""
    return f"问题: {question}\n让我们一步一步地思考，来回答这个问题。"


def user_with_knowledge(question: str, knowledge: str) -> str:
    """Chain-of-thought conditioned on regulation text.

    Used both for the oracle setting (gold regulations) and the RAG setting
    (retrieved passages) -- the wording is identical, only the source of
    ``knowledge`` differs.
    """
    return (
        f"问题: {question}\n相关的知识: {knowledge}\n"
        "以上的知识也许对回答问题有帮助。让我们一步一步地思考，来回答这个问题。"
    )


def user_pot(question: str) -> str:
    """Program-of-Thought: ask the model for a ``solution()`` function."""
    return (
        f"问题: {question}\n\n根据给定问题生成一个Python程序。从以下代码开始：\n"
        "'''python\ndef solution():\n    # 定义变量（注释说明每个变量的税务含义）\n"
    )


#: Settings whose prompt is a single turn conditioned on regulation text. They
#: share one prompt; only the provenance of ``knowledge`` differs (gold
#: annotations for oracle, dense retrieval for standard, summarised retrieval
#: for summary).
KNOWLEDGE_MODES = ("oracle", "rag", "standard", "summary")


def build_messages(mode: str, question: str, knowledge: str | None = None) -> list[dict]:
    """Assemble the chat messages for one single-turn example.

    ``mode`` is ``cot`` (closed book), ``pot`` (generate a Python program), or
    one of :data:`KNOWLEDGE_MODES` (conditioned on ``knowledge``). The
    multi-turn settings (``interact``, ``inlinesearch``) drive their own
    conversations -- see :mod:`taxreasoning.methods`.
    """
    if mode == "cot":
        return [
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": user_cot(question)},
        ]
    if mode in KNOWLEDGE_MODES:
        if knowledge is None:
            raise ValueError(f"mode={mode!r} requires knowledge text")
        return [
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": user_with_knowledge(question, knowledge)},
        ]
    if mode in ("pot", "pal"):  # "pal" kept as an alias for older result files
        return [
            {"role": "system", "content": SYSTEM_PAL},
            {"role": "user", "content": user_pot(question)},
        ]
    raise ValueError(f"unknown mode {mode!r}")


# --- LLM-as-judge ------------------------------------------------------------

JUDGE_SYSTEM_ANSWER = "你是一个模型输出评估助手。"

JUDGE_SYSTEM_PROGRAM = "你是模型输出评估助手，只返回1或0。"


def judge_answer(gold: str, prediction: str) -> list[dict]:
    """Judge a free-text chain-of-thought answer against the gold answer."""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_ANSWER},
        {
            "role": "user",
            "content": (
                f"真实答案 (含数值和单位)：\n{gold}\n\n"
                f"模型完整输出（含推理过程及最终返回值）：\n{prediction}\n\n"
                "请判断“模型输出的最终答案”是否与“真实答案”完全一致。"
                "仅返回True或False，不要其他多余文字。"
            ),
        },
    ]


def judge_program(gold: str, run_answer) -> list[dict]:
    """Judge the value returned by an executed ``solution()`` function."""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROGRAM},
        {
            "role": "user",
            "content": (
                f"真实答案：{gold}\n模型运行结果：{run_answer}\n"
                "请仅返回1（一致）或0（不一致)。"
            ),
        },
    ]
