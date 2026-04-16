from langchain_classic.schema import HumanMessage, SystemMessage

from graph.state import SelfRAGState
from prompts.prompts import REVISE_ANSWER_PROMPT


def revise_answer(state: SelfRAGState, llm, app_config: dict) -> dict:
    try:
        prompt = REVISE_ANSWER_PROMPT.format(
            context=state.get("context", ""),
            answer=state.get("answer", ""),
        )
        response = llm.invoke([
            SystemMessage(content="You are an answer revision system."),
            HumanMessage(content=prompt),
        ])
        return {
            "answer": response.content,
            "revision_count": state.get("revision_count", 0) + 1,
        }
    except Exception:
        return {"revision_count": state.get("revision_count", 0) + 1}
