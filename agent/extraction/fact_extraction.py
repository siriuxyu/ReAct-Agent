from __future__ import annotations


async def extract_session_observations(session_text: str, model: str) -> str:
    """
    Extract concise factual statements from a session transcript.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.utils import load_chat_model

    llm = load_chat_model(model)
    messages = [
        SystemMessage(
            content=(
                "Extract concise factual statements from the conversation below. "
                "Output one fact per line. Include specific details like names, dates, "
                "events, relationships. Do not include opinions or uncertain information. "
                "Example output:\n"
                "- Caroline is a transgender woman\n"
                "- Melanie has two kids and works full time\n"
                "- Caroline joined an LGBTQ support group on May 7, 2023"
            )
        ),
        HumanMessage(content=f"Conversation:\n{session_text}\n\nFacts:"),
    ]
    response = await llm.ainvoke(messages)
    return response.content.strip()
