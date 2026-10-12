import json

import pytest

from guardrail_proxy.judge import JudgeError, LLMJudge, split_sentences
from guardrail_proxy.providers import FakeLLM

GOOD = {
    "claims": [
        {"claim": "Start at 500 mg", "verdict": "supported", "evidence": "500 mg"},
        {"claim": "Safe in severe renal impairment", "verdict": "contradicted", "evidence": "x"},
        {"claim": "Prevents stroke", "verdict": "not_found", "evidence": ""},
        {"claim": "Take with meals", "verdict": "supported", "evidence": "with meals"},
    ]
}


async def test_llm_judge_parses_structured_output():
    llm = FakeLLM(lambda msgs: json.dumps(GOOD))
    j = await LLMJudge(llm, "judge").judge("answer", "context")
    assert len(j.claims) == 4 and j.risk == 0.5
    assert [c.claim for c in j.unsupported] == [
        "Safe in severe renal impairment",
        "Prevents stroke",
    ]


async def test_llm_judge_tolerates_think_blocks_and_fences():
    raw = "<think>hmm</think>\n```json\n" + json.dumps(GOOD) + "\n```"
    j = await LLMJudge(FakeLLM(lambda m: raw), "judge").judge("a", "c")
    assert len(j.claims) == 4


async def test_llm_judge_retries_then_fails():
    llm = FakeLLM(lambda m: "not json")
    with pytest.raises(JudgeError):
        await LLMJudge(llm, "judge").judge("a", "c")
    assert len(llm.calls) == 2


def test_sentence_split():
    assert split_sentences("Take 500 mg daily. Do not crush. <PERSON_1> agreed.") == [
        "Take 500 mg daily.",
        "Do not crush.",
        "<PERSON_1> agreed.",
    ]
