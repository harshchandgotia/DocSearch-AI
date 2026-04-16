from typing import Literal

from pydantic import BaseModel
from langchain_classic.schema import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from prompts.prompts import HALLUCINATION_CHECK_PROMPT


class HallucinationCheck(BaseModel):
    support: Literal["fully_supported", "partially_supported", "not_supported"]


def check_hallucination(state: SelfRAGState, llm, app_config: dict) -> dict:
    try:
        prompt = HALLUCINATION_CHECK_PROMPT.format(
            question=state["original_question"],
            context=state.get("context", ""),
            answer=state.get("answer", ""),
        )
        structured_llm = llm.with_structured_output(HallucinationCheck)
        result = structured_llm.invoke([
            SystemMessage(content="You are a hallucination detection system."),
            HumanMessage(content=prompt),
        ])
        return {"is_supported": result.support}
    except Exception:
        return {"is_supported": "not_supported"}
