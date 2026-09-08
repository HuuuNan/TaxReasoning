"""The knowledge-augmentation strategies evaluated in the paper.

Table 2 of the paper reports seven methods. Four of them need machinery beyond
a single prompt, and live here:

``standard``
    Single-turn. Top-5 documents from the dense retriever, dropped into the
    same prompt the oracle setting uses.
``summary``
    Single-turn. The retrieved documents are summarised by GPT-4o and filtered
    for relevance, "to mitigate long-context forgetting".
``interact``
    Multi-turn. The model sees summaries of the top-10 documents and issues
    ``Check: Document [i]`` / ``Output:`` / ``End.`` actions.
``inlinesearch``
    Multi-turn. The model issues ``Search: query`` / ``Output:`` / ``End.``
    actions, retrieving on demand mid-reasoning.

The two multi-turn protocols follow ALCE (Gao et al. 2023), which the paper
cites for them; the action vocabulary is taken from ALCE's
``*_interact_doc_id`` and ``*_interact_search`` prompts, translated to this
task (Chinese tax questions, a numeric final answer, no citation markers).

The ``cot``, ``pot`` and ``oracle`` settings need no retrieval and are built
directly by :mod:`taxreasoning.prompts`.
"""

from __future__ import annotations

import logging
import re

from taxreasoning import llm, prompts

logger = logging.getLogger(__name__)

DEFAULT_TOP_K_SINGLE = 5    # "the top-5 most relevant tax documents"
DEFAULT_TOP_K_MULTI = 10    # "starts with summaries of the top-10 documents"
DEFAULT_MAX_TURNS = 8

# --- Summary -----------------------------------------------------------------

SUMMARIZE_SYSTEM = (
    "你是一个税法资料整理助手。给定一个税务计算问题和一篇税法文件，"
    "判断该文件对解答问题是否相关，并按以下格式回复：\n"
    "若不相关，回复：不相关\n"
    "若相关，回复：相关。<摘要>\n"
    "摘要须保留解题所需的全部关键信息：适用条件、税率表、速算扣除数、"
    "计算公式、例外条款和金额门槛。不要省略数字，不要作任何计算。"
)

# Kept byte-identical to what the original rag/ scripts consumed, so existing
# refine.json dumps stay readable: an entry either starts with "不相关" or with
# "相关。" followed by the summary.
IRRELEVANT_PREFIX = "不相关"
RELEVANT_PREFIX = "相关。"


def summarize_document(client, model: str, question: str, document: str) -> str:
    """Summarise one retrieved document, or mark it irrelevant."""
    return llm.chat(
        client,
        model,
        [
            {"role": "system", "content": SUMMARIZE_SYSTEM},
            {"role": "user", "content": f"问题: {question}\n\n税法文件:\n{document}"},
        ],
        temperature=0,
    ).strip()


def build_summary_knowledge(client, model: str, question: str, documents: list[dict]) -> tuple[str, list[dict]]:
    """Summarise and relevance-filter retrieved documents.

    Returns ``(knowledge, per_document_records)``. If every document is judged
    irrelevant the top-ranked one is kept anyway, matching the fallback in the
    original scripts -- an empty context would make the setting indistinguishable
    from closed-book CoT.
    """
    kept, records = [], []
    for doc in documents:
        verdict = summarize_document(client, model, question, doc["contents"])
        relevant = not verdict.startswith(IRRELEVANT_PREFIX)
        summary = verdict[len(RELEVANT_PREFIX):] if verdict.startswith(RELEVANT_PREFIX) else verdict
        records.append({"id": doc["id"], "title": doc["title"], "relevant": relevant, "refine": verdict})
        if relevant:
            kept.append(summary.strip())

    if not kept and documents:
        logger.debug("all documents judged irrelevant; falling back to the top-ranked one")
        kept = [documents[0]["contents"]]

    return "\n\n".join(kept), records


# --- Multi-turn action parsing -----------------------------------------------

CHECK_RE = re.compile(r"Check:\s*Document\s*((?:\[\d+\]\s*)+)", re.IGNORECASE)
SEARCH_RE = re.compile(r"Search:\s*(.+)", re.IGNORECASE)
OUTPUT_RE = re.compile(r"Output:\s*(.*)", re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"\bEnd\b\.?", re.IGNORECASE)

INTERACT_SYSTEM = (
    "你作为资深税务会计师，需解决以下税法问题。你会先看到若干税法文件的摘要。\n"
    "你可以执行以下三种动作之一，每次回复只执行一个动作：\n"
    '1. "Check: Document [i]" —— 查看第 i 篇文件的完整内容（一次最多查看 3 篇，'
    "只查看你认为相关的文件）；\n"
    '2. "Output: <推理>" —— 基于已获得的信息写出推理步骤；\n'
    '3. "End." —— 结束。\n'
    "在最后一次 Output 中，你需要总结你的回答，并以“因此，答案是{最终答案}”结尾。"
    "最终答案应为一个数值。"
)

INLINESEARCH_SYSTEM = (
    "你作为资深税务会计师，需解决以下税法问题。你可以在推理过程中随时检索税法资料。\n"
    "你可以执行以下三种动作之一，每次回复只执行一个动作：\n"
    '1. "Search: <检索词>" —— 用检索词查找最相关的税法文件全文；\n'
    '2. "Output: <推理>" —— 基于已获得的信息写出推理步骤；\n'
    '3. "End." —— 结束。\n'
    "在最后一次 Output 中，你需要总结你的回答，并以“因此，答案是{最终答案}”结尾。"
    "最终答案应为一个数值。"
)


def format_documents(documents: list[dict], field: str = "contents") -> str:
    """Render documents as ``Document [i](Title: ...): ...``, as ALCE does."""
    return "\n\n".join(
        f"Document [{i + 1}](Title: {doc['title']}): {doc.get(field, doc['contents'])}"
        for i, doc in enumerate(documents)
    )


def run_interact(
    client,
    model: str,
    question: str,
    documents: list[dict],
    summaries: list[str],
    max_turns: int = DEFAULT_MAX_TURNS,
) -> tuple[str, list[dict]]:
    """Multi-turn loop where the model reveals full documents on demand.

    ``summaries[i]`` is what the model sees for ``documents[i]`` up front;
    a ``Check`` action swaps in the full text. Returns the concatenated
    ``Output:`` segments and a trace of the actions taken.
    """
    shown = [dict(doc, contents=summary) for doc, summary in zip(documents, summaries)]
    messages = [
        {"role": "system", "content": INTERACT_SYSTEM},
        {"role": "user", "content": f"问题: {question}\n\n{format_documents(shown)}"},
    ]
    return _action_loop(client, model, messages, documents, max_turns, retriever=None)


def run_inlinesearch(
    client,
    model: str,
    question: str,
    retriever,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> tuple[str, list[dict]]:
    """Multi-turn loop where the model issues its own retrieval queries."""
    messages = [
        {"role": "system", "content": INLINESEARCH_SYSTEM},
        {"role": "user", "content": f"问题: {question}"},
    ]
    return _action_loop(client, model, messages, documents=[], max_turns=max_turns, retriever=retriever)


def _action_loop(client, model, messages, documents, max_turns, retriever):
    """Shared driver for the two multi-turn protocols."""
    outputs: list[str] = []
    trace: list[dict] = []

    for turn in range(max_turns):
        reply = llm.chat(client, model, messages).strip()
        messages.append({"role": "assistant", "content": reply})

        check = CHECK_RE.search(reply)
        search = SEARCH_RE.search(reply)
        output = OUTPUT_RE.search(reply)

        # Order matters: a reply may mention several actions, and the protocol
        # says one per turn. Retrieval actions take precedence over Output so a
        # model that asks for a document and then guesses still gets the
        # document before committing.
        if check:
            ids = [int(n) - 1 for n in re.findall(r"\[(\d+)\]", check.group(1))][:3]
            revealed = [documents[i] for i in ids if 0 <= i < len(documents)]
            trace.append({"turn": turn, "action": "check", "doc_ids": ids})
            content = format_documents(revealed) if revealed else "没有找到对应编号的文件。"
            messages.append({"role": "user", "content": content})
            continue

        if search and retriever is not None:
            query = search.group(1).strip().split("\n")[0]
            results = retriever.search(query, top_k=1)
            trace.append({"turn": turn, "action": "search", "query": query,
                          "retrieved": [r["title"] for r in results]})
            content = format_documents(results) if results else "没有检索到相关文件。"
            messages.append({"role": "user", "content": content})
            continue

        if output:
            # OUTPUT_RE is DOTALL and greedy, so a trailing "End." on its own
            # line lands inside the captured text. Split it off rather than
            # looking after the match, or the loop never sees the terminator
            # and burns every remaining turn.
            text = output.group(1).strip()
            terminator = END_RE.search(text)
            if terminator:
                text = text[: terminator.start()].strip()
            if text:
                outputs.append(text)
            trace.append({"turn": turn, "action": "output", "chars": len(text)})
            if terminator:
                trace.append({"turn": turn, "action": "end"})
                break
            messages.append({"role": "user", "content": "继续。"})
            continue

        if END_RE.search(reply):
            trace.append({"turn": turn, "action": "end"})
            break

        # No recognisable action -- treat the whole reply as reasoning so a
        # model that ignores the protocol still produces a scorable answer.
        outputs.append(reply)
        trace.append({"turn": turn, "action": "unparsed", "chars": len(reply)})
        break

    else:
        trace.append({"turn": max_turns, "action": "max_turns_reached"})

    return "\n".join(outputs), trace


def build_standard_messages(question: str, documents: list[dict]) -> list[dict]:
    """Single-turn prompt over raw retrieved documents."""
    knowledge = "\n\n".join(doc["contents"] for doc in documents)
    return prompts.build_messages("rag", question, knowledge)
