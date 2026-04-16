from typing import Literal

from pydantic import BaseModel
from langchain_classic.schema import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from prompts.prompts import USEFULNESS_CHECK_PROMPT


class UsefulnessCheck(BaseModel):
    is_useful: Literal["useful", "not_useful"]
    reason: str


def check_usefulness(state: SelfRAGState, llm, app_config: dict) -> dict:
    try:
        prompt = USEFULNESS_CHECK_PROMPT.format(
            question=state["original_question"],
            answer=state.get("answer", ""),
        )
        structured_llm = llm.with_structured_output(UsefulnessCheck)
        result = structured_llm.invoke([
            SystemMessage(content="You are a usefulness evaluation system."),
            HumanMessage(content=prompt),
        ])
        return {"is_useful": result.is_useful}
    except Exception:
        return {"is_useful": "useful"}
